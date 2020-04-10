import os
import torch
import tabulate
import torch.nn as nn
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('svg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})

class Hook_record_input():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input

    def close(self):
        self.hook.remove()

class Hook_record_grad():
    def __init__(self, module):
        self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, grad_input, grad_output):
        self.grad_output = grad_output

    def close(self):
        self.hook.remove()

class Hook_sparsify_grad_input():
    def __init__(self, module, gamma=0.5):
        self.hook = module.register_backward_hook(self.hook_fn)
        self.gamma = gamma

    def hook_fn(self, module, grad_input, grad_output):
        num_grad_input_to_keep = int(grad_input[0].numel() * self.gamma) # grad_input contains grad for input, weight and bias
        threshold, _ = torch.kthvalue(abs(grad_input[0]).view(-1), num_grad_input_to_keep)
        grad_input_new = grad_input[0]
        grad_input_new[abs(grad_input[0]) < threshold] = 0
        #self.grad_input = grad_input_new
        return (grad_input_new, grad_input[1], grad_input[2])

    def close(self):
        self.hook.remove()


def add_input_record_Hook(model, name_as_key=False):
    Hooks = {}
    if name_as_key:
        for name,module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                Hooks[name] = Hook_record_input(module)
            
    else:
        for k,module in enumerate(model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                Hooks[k] = Hook_record_input(module)
    return Hooks

def add_grad_record_Hook(model, name_as_key=False):
    Hooks = {}
    if name_as_key:
        for name,module in model.named_modules():
            Hooks[name] = Hook_record_grad(module)
            
    else:
        for k,module in enumerate(model.modules()):
            Hooks[k] = Hook_record_grad(module)
    return Hooks

def add_sparsify_grad_input_Hook(model, gamma=0.5, name_as_key=False):
    Hooks = {}
    if name_as_key:
        for name,module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d): # only sparsify grad_input of batchnorm, which is grad_output of conv2d
                Hooks[name] = Hook_sparsify_grad_input(module, gamma)
            elif isinstance(module, nn.Conv2d):
                Hooks[name] = Hook_record_grad(module)
            
    else:
        for k,module in enumerate(model.modules()):
            if isinstance(module, nn.BatchNorm2d): # only sparsify grad_input of batchnorm, which is grad_output of conv2d
                Hooks[k] = Hook_sparsify_grad_input(module, gamma)
            elif isinstance(module, nn.Conv2d):
                Hooks[k] = Hook_record_grad(module)
    return Hooks

def remove_hooks(Hooks):
    for k in Hooks.keys():
        Hooks[k].close()

def getBatchNormBiasRegularizer(model):
    Loss = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            Loss += torch.exp(m.bias).sum()
    return Loss

def set_BN_bias(model, init_bias):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.bias.data = init_bias * torch.ones_like(m.bias.data)

def freeze_BN_layers(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            m.track_running_stats = False

def get_weight_sparsity(model):
    total = 0.
    nonzeros = 0.
    for m in model.modules():
        if hasattr(m, 'mask_keep_original'):
            nonzeros += m.mask_keep_original.sum()
            total += m.mask_keep_original.numel()
    return 0 if total == 0 else ((total - nonzeros) / total).cpu().numpy().item()

def get_activation_sparsity(Hooks):
    total = 0.
    nonzeros = 0.
    for k in Hooks.keys():
        input = Hooks[k].input[0]
        input_mask = (input != 0).float()
        nonzeros += input_mask.sum()
        total += input_mask.numel()
    input_sparsity = (total - nonzeros) / total
    return input_sparsity

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)


#@profile
def run_epoch(loader, model, criterion, optimizer=None,
              phase="train", loss_scaling=1.0, lambda_BN=0.0):
    assert phase in ["train", "val", "test"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase=="train": model.train()
    elif phase=="val" or phase=="test": model.eval()

    ttl = 0
    #Hooks = add_input_record_Hook(model)
    #Hooks_grad = add_grad_record_Hook(model)
    with torch.autograd.set_grad_enabled(phase=="train"):
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            if lambda_BN > 0:
                loss += lambda_BN * getBatchNormBiasRegularizer(model)

            loss_sum += loss * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += input.size()[0]

            if phase=="train":
                optimizer.zero_grad()
                loss = loss * loss_scaling # grad scaling
                loss.backward()
                optimizer.step()

    correct = correct.cpu().item()
    return {
        'loss': loss_sum.cpu().item() / float(ttl),
        'accuracy': correct / float(ttl) * 100.0,
    }

def log2df(log_file_name):
    '''
    return a pandas dataframe from a log file
    '''
    with open(log_file_name, 'r') as f:
        lines = f.readlines() 
    # search backward to find table header
    num_lines = len(lines)
    for i in range(num_lines):
        if lines[num_lines-1-i].startswith('---'):
            break
    header_line = lines[num_lines-2-i]
    num_epochs = i
    columns = header_line.split()
    df = pd.DataFrame(columns=columns)
    for i in range(num_epochs):
        df.loc[i] = [float(x) for x in lines[num_lines-num_epochs+i].split()]
    return df 

def plot_data_dict(data_dict_list, result_file_name, xlabel='x', ylabel='y', yscale='auto', xlim=None, ylim=None):
    # change figure file type
    #filetype = result_file_name.split('.')[-1]
    #matplotlib.use(filetype)

    # define matplotlib parameters
    markers = ['s','D','X','v','^','P','X', 'p', 'o']
    fig = plt.figure(figsize=(10,8), dpi=300)
    ax = fig.add_subplot(1,1,1)
    ymin = float('Inf')
    ymax = float('-Inf')    
    
    if isinstance(data_dict_list, dict):
        data_dict_list = [data_dict_list]
    k = 0
    for data_dict in data_dict_list:
        marker = markers[k % 8]
        y = np.asarray(data_dict['y'])
        if 'x' in data_dict.keys():
            x = np.asarray(data_dict['x'])
        else:
            x = np.array([x for x in range(len(y))])
        if 'label' not in data_dict:
            data_dict['label'] = 'exp %d' % k
        # check how many non-nan values
        count = 0
        for y_value in y:
            if np.isnan(y_value) == False:
                count += 1
        markevery = int(np.ceil(count / 40.))
        ax.plot(x, y, marker=marker, label=data_dict['label'], alpha=0.7,
            markevery=markevery)
        if y.max() > ymax:
            ymax = y.max()
        if y.min() < ymin:
            ymin = y.min()
        k += 1
    if len(data_dict_list) > 10:
        ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5), fontsize=16)
    else:
        ax.legend(loc='best')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yscale == 'log' or (yscale == 'auto' and (abs(ymax/(ymin+1e-8)) >= 100 and ymin > 0)):
        ax.set_yscale('log')
    plt.grid()
    plt.savefig(result_file_name, bbox_inches='tight')


if __name__ == "__main__":
    log2df('logs/VGG16BN_FP8_TD_4_0.0_0.0_0.9375_0.99_5.0.log')
