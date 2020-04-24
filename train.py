import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import tabulate
import models
from data import get_data
import numpy as np
from qtorch.auto_low import lower
from qtorch.optim import OptimLP
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.quant import quantizer, Quantizer
import logging
torch.manual_seed(0)
np.random.seed(0)

num_types = ["weight", "activate", "grad", "error", "momentum", "acc"]

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name: CIFAR10 or IMAGENET12')
parser.add_argument('--data_path', type=str, default="./data",
                    help='path to datasets location (default: "./data")')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size (default: 128)')
parser.add_argument('--val_ratio', type=float, default=0.0,
                    help='Ratio of the validation set (default: 0.0)')
parser.add_argument('--num_workers', type=int, default=2, 
                    help='number of workers (default: 2)')
parser.add_argument('--model', type=str, default='VGG16',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, 
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--epochs', type=int, default=200, 
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr_init', type=float, default=0.01,
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--log_file', type=str, default=None,
                    help='path to log file')
parser.add_argument('--save_file', type=str, default=None,
                    help='path to save file for model')
parser.add_argument('--TD_gamma', type=float, default=0.0,
                    help='gamma value for targeted dropout')
parser.add_argument('--TD_alpha', type=float, default=0.0,
                    help='alpha value for targeted dropout')
parser.add_argument('--block_size', type=int, default=16,
                    help='block size for dropout')
parser.add_argument('--evaluate', type=str, default=None,
                    help='model file for accuracy evaluation')
parser.add_argument('--TD_gamma_final', type=float, default=-1.0,
                    help='final gamma value for targeted dropout')
parser.add_argument('--TD_alpha_final', type=float, default=-1.0,
                    help='final alpha value for targeted dropout')
parser.add_argument('--ramping_power', type=float, default=3.0,
                    help='power of ramping schedule')
parser.add_argument('--lambda_BN', type=float, default=0,
                    help='lambda for BN bias regularization')
parser.add_argument('--init_BN_bias', type=float, default=0,
                    help='initial bias for batch norm')
parser.add_argument('--gradient_gamma', type=float, default=0.0,
                    help='prunning ratio for gradient during backward')
parser.add_argument('--freeze_BN_after', type=int, default=10000, # not helpful
                    help='epoch number after which BN parameters freeze')
parser.add_argument('--per_layer', type=int, default=0,
                    help='uniform pruning rate across layers if 1')


for num in num_types:
    parser.add_argument('--{}-man'.format(num), type=int, default=-1, metavar='N',
                        help='number of bits to use for mantissa of {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-exp'.format(num), type=int, default=-1, metavar='N',
                        help='number of bits to use for exponent of {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-rounding'.format(num), type=str, default='stochastic', metavar='S',
                        choices=["stochastic","nearest"],
                        help='rounding method for {}, stochastic or nearest'.format(num))

args = parser.parse_args()
logger = logging.getLogger('training')
if args.log_file is not None:
    fileHandler = logging.FileHandler(args.log_file)
    fileHandler.setLevel(0)
    logger.addHandler(fileHandler)
streamHandler = logging.StreamHandler()
streamHandler.setLevel(0)
logger.addHandler(streamHandler)
logger.root.setLevel(0)

logger.info(args)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

loaders = get_data(args.dataset, args.data_path, args.batch_size, args.val_ratio, args.num_workers)
if args.dataset=="CIFAR10": num_classes=10
elif args.dataset=="IMAGENET12": num_classes=1000

def get_result(loaders, model, phase, loss_scaling=1000.0, lambda_BN=0.0):
    time_ep = time.time()
    res = utils.run_epoch(loaders[phase], model, criterion,
                                optimizer=optimizer, phase=phase, loss_scaling=loss_scaling,
                                lambda_BN=lambda_BN)
    time_pass = time.time() - time_ep
    res['time_pass'] = time_pass
    return res

if args.evaluate is not None:
    checkpoint = torch.load(args.evaluate)
    # replace certain args with saved args
    saved_args = checkpoint['args']
    for num in num_types:
        setattr(args, '{}_man'.format(num), getattr(saved_args, '{}_man'.format(num)))
        setattr(args, '{}_exp'.format(num), getattr(saved_args, '{}_exp'.format(num)))
        setattr(args, '{}_rounding'.format(num), getattr(saved_args, '{}_rounding'.format(num)))
    args.block_size = saved_args.block_size
    args.model = saved_args.model
    args.TD_alpha = 1


if 'LP' in args.model:
    quantizers = {}
    for num in num_types:
        num_rounding = getattr(args, "{}_rounding".format(num))
        num_man = getattr(args, "{}_man".format(num))
        num_exp = getattr(args, "{}_exp".format(num))
        number = FloatingPoint(exp=num_exp, man=num_man)
        logger.info("{}: {} rounding, {}".format(num, num_rounding,
                                           number))
        quantizers[num] = quantizer(forward_number=number, forward_rounding=num_rounding)
# Build model
model_cfg = getattr(models, args.model)
if 'LP' in args.model:
    activate_number = FloatingPoint(exp=args.activate_exp, man=args.activate_man)
    error_number = FloatingPoint(exp=args.error_exp, man=args.error_man)
    logger.info("activation: {}, {}".format(args.activate_rounding, activate_number))
    logger.info("error: {}, {}".format(args.error_rounding, error_number))
    make_quant = lambda : Quantizer(activate_number, error_number, args.activate_rounding, args.error_rounding)
    #make_quant = nn.Identity
    model_cfg.kwargs.update({"quant":make_quant})

if 'TD' in args.model:
    logger.info("block size: {}".format(args.block_size))
    logger.info("TD gamma: {0:.3f}".format(args.TD_gamma))
    logger.info("TD alpha: {0:.3f}".format(args.TD_alpha))
    model_cfg.kwargs.update({"gamma":args.TD_gamma, "alpha":args.TD_alpha, "block_size":args.block_size})

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
logger.info('Model: {}'.format(model))
model.cuda()
criterion = F.cross_entropy
optimizer = SGD(
   model.parameters(),
   lr=args.lr_init,
   momentum=args.momentum,
   weight_decay=args.wd,
)
loss_scaling = 1.0

if args.init_BN_bias != 0:
    utils.set_BN_bias(model, args.init_BN_bias)

if 'LP' in args.model:
    loss_scaling = 1000.0
    optimizer = OptimLP(optimizer,
                        weight_quant=quantizers["weight"],
                        grad_quant=quantizers["grad"],
                        momentum_quant=quantizers["momentum"],
                        acc_quant=quantizers["acc"],
                        grad_scaling=1/loss_scaling # scaling down the gradient
    )

Hooks_input = utils.add_input_record_Hook(model)
if args.evaluate is not None:
    model.load_state_dict(checkpoint['state_dict'])
    # update TD gamma and alpha value if needed
    for m in model.modules():
        if hasattr(m, 'gamma'):
            m.gamma = args.TD_gamma
            m.alpha = args.TD_alpha
    print(model)
    test_res = get_result(loaders, model, "test", loss_scaling=1)
    print("test accuracy = %.3f%%" % test_res['accuracy'])
    print("Weight sparsity = %.3f%%" % (utils.get_weight_sparsity(model) * 100.))
    print("Activation sparsity = %.3f%%" % (utils.get_activation_sparsity(Hooks_input) * 100.))
    print("Activation group sparsity = %.3f%%" % (utils.get_activation_group_sparsity(Hooks_input) * 100.))
    exit()
if args.gradient_gamma > 0:
    Hooks_sparsify_grad = utils.add_sparsify_grad_input_Hook(model, args.gradient_gamma)

def schedule(epoch):
    t = (epoch) / args.epochs
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio

    return factor

def update_gamma_alpha(epoch):
    if args.TD_gamma_final > 0:
        TD_gamma = args.TD_gamma_final - (((args.epochs - 1 - epoch)/(args.epochs - 1)) ** args.ramping_power) * (args.TD_gamma_final - args.TD_gamma)
        for m in model.modules():
            if hasattr(m, 'gamma'):
                m.gamma = TD_gamma
    else:
        TD_gamma = args.TD_gamma
    if args.TD_alpha_final > 0:
        TD_alpha = args.TD_alpha_final - (((args.epochs - 1 - epoch)/(args.epochs - 1)) ** args.ramping_power) * (args.TD_alpha_final - args.TD_alpha)
        for m in model.modules():
            if hasattr(m, 'alpha'):
                m.alpha = TD_alpha
    else:
        TD_alpha = args.TD_alpha
    return TD_gamma, TD_alpha

scheduler = LambdaLR(optimizer, lr_lambda=[schedule])
# Prepare logging
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_time', 'te_loss', 'te_acc', 'te_time', 'wspar', 'aspar', 'agspar']
if args.TD_gamma_final > 0 or args.TD_alpha_final > 0:
    columns += ['gamma', 'alpha']
    
for epoch in range(args.epochs):
    time_ep = time.time()
    TD_gamma, TD_alpha = update_gamma_alpha(epoch)
    utils.update_mask(model, TD_gamma, TD_alpha, args.block_size, args.per_layer)
    train_res = get_result(loaders, model, "train", loss_scaling, args.lambda_BN)
    test_res = get_result(loaders, model, "test", loss_scaling)
    scheduler.step()
    weight_sparsity = utils.get_weight_sparsity(model)
    activation_sparsity = utils.get_activation_sparsity(Hooks_input)
    activation_group_sparsity = utils.get_activation_group_sparsity(Hooks_input)

    values = [epoch + 1, optimizer.param_groups[0]['lr'], train_res['loss'], train_res['accuracy'], 
            train_res['time_pass'], test_res['loss'], test_res['accuracy'], test_res['time_pass'], weight_sparsity, activation_sparsity, activation_group_sparsity]
    if args.TD_gamma_final > 0 or args.TD_alpha_final > 0:
        values += [TD_gamma, TD_alpha]
    utils.print_table(values, columns, epoch, logger)
    if epoch == args.freeze_BN_after:
        utils.freeze_BN_layers(model)
        Hooks_grad = utils.add_grad_record_Hook(model)


if args.save_file is not None:
    torch.save({
        'state_dict': model.state_dict(),
        'args': args}, 
        os.path.join('checkpoint', args.save_file))
