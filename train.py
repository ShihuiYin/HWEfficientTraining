import argparse
import os
import sys
import time
import torch
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
parser.add_argument('--num_workers', type=int, default=4, 
                    help='number of workers (default: 4)')
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

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

loaders = get_data(args.dataset, args.data_path, args.batch_size, args.val_ratio, args.num_workers)
if args.dataset=="CIFAR10": num_classes=10
elif args.dataset=="IMAGENET12": num_classes=1000

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
if 'LP' in args.model:
    loss_scaling = 1000.0
    optimizer = OptimLP(optimizer,
                        weight_quant=quantizers["weight"],
                        grad_quant=quantizers["grad"],
                        momentum_quant=quantizers["momentum"],
                        acc_quant=quantizers["acc"],
                        grad_scaling=1/loss_scaling # scaling down the gradient
    )

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

scheduler = LambdaLR(optimizer, lr_lambda=[schedule])
# Prepare logging
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_time', 'te_loss', 'te_acc', 'te_time']

def get_result(loaders, model, phase, loss_scaling=1000.0):
    time_ep = time.time()
    res = utils.run_epoch(loaders[phase], model, criterion,
                                optimizer=optimizer, phase=phase, loss_scaling=loss_scaling)
    time_pass = time.time() - time_ep
    res['time_pass'] = time_pass
    return res

if args.evaluate is not None:
    model = torch.load(args.evaluate).cuda()
    # update TD gamma and alpha value if needed
    for m in model.modules():
        if hasattr(m, 'gamma'):
            m.gamma = args.TD_gamma
            m.alpha = args.TD_alpha
    print(model)
    test_res = get_result(loaders, model, "test", loss_scaling)
    print("test accuracy = %.3f%%" % test_res['accuracy'])
    exit()

for epoch in range(args.epochs):

    scheduler.step()
    time_ep = time.time()
    train_res = get_result(loaders, model, "train", loss_scaling)
    test_res = get_result(loaders, model, "test", loss_scaling)

    values = [epoch + 1, optimizer.param_groups[0]['lr'], train_res['loss'], train_res['accuracy'], train_res['time_pass'], test_res['loss'], test_res['accuracy'], test_res['time_pass']]

    utils.print_table(values, columns, epoch, logger)

if args.save_file is not None:
    torch.save(model,  os.path.join('checkpoint', args.save_file))
