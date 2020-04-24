import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d_TD(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, gamma=0.0, alpha=0.0, block_size=16):
        super(Conv2d_TD, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.gamma = gamma
        self.alpha = alpha
        self.block_size = block_size
        self.count = -1 # disable in-module mask generation
    
    def forward(self, input):
        # sort blocks by mean absolute value
        if self.gamma > 0 and self.alpha > 0:
            if self.count % 1000 == 0:
                with torch.no_grad():
                    block_values = F.avg_pool2d(self.weight.data.abs().permute(2,3,0,1),
                                    kernel_size=(self.block_size, self.block_size),
                                    stride=(self.block_size, self.block_size))
                    sorted_block_values, indices = torch.sort(block_values.contiguous().view(-1))
                    thre_index = int(block_values.data.numel() * self.gamma)
                    threshold = sorted_block_values[thre_index]
                    mask_small = 1 - block_values.gt(threshold.cuda()).float().cuda() # mask for blocks candidates for pruning
                    mask_dropout = torch.rand_like(block_values).lt(self.alpha).float().cuda()
                    mask_keep = 1.0 - mask_small * mask_dropout
                    self.mask_keep_original = F.interpolate(mask_keep, 
                                        scale_factor=(self.block_size, self.block_size)).permute(2,3,0,1)
            # self.count += 1 # disable in-module mask_generation
            out = F.conv2d(input, self.weight * self.mask_keep_original, None, self.stride, self.padding,
                                        self.dilation, self.groups)
        else:
            out = F.conv2d(input, self.weight, None, self.stride, self.padding,
                                        self.dilation, self.groups)
        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
    
    def extra_repr(self):
        return super(Conv2d_TD, self).extra_repr() + ', gamma={}, alpha={}, block_size={}'.format(
                self.gamma, self.alpha, self.block_size)

class Linear_TD(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, gamma=0.0, alpha=0.0, block_size=16):
        super(Linear_TD, self).__init__(in_features, out_features, bias)
        self.gamma = gamma
        self.alpha = alpha
        self.block_size = block_size
        self.count = -1 # disable in-module mask generation

    def forward(self, input):
        if self.gamma > 0 and self.alpha > 0:
            if self.count % 1000 == 0:
                with torch.no_grad():
                    block_values = F.avg_pool2d(self.weight.data.abs().unsqueeze(0),
                                    kernel_size=(self.block_size, self.block_size),
                                    stride=(self.block_size, self.block_size))
                    sorted_block_values, indices = torch.sort(block_values.contiguous().view(-1))
                    thre_index = int(block_values.data.numel() * self.gamma)
                    threshold = sorted_block_values[thre_index]
                    mask_small = 1 - block_values.gt(threshold.cuda()).float().cuda() # mask for blocks candidates for pruning
                    mask_dropout = torch.rand_like(block_values).lt(self.alpha).float().cuda()
                    mask_keep = 1.0 - mask_small * mask_dropout
                    self.mask_keep_original = F.interpolate(mask_keep.unsqueeze(0), 
                                        scale_factor=(self.block_size, self.block_size)).squeeze()
            # self.count += 1 # disable in-module mask generation
            return F.linear(input, self.weight * self.mask_keep_original, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return super(Linear_TD, self).extra_repr() + ', gamma={}, alpha={}, block_size={}'.format(
                self.gamma, self.alpha, self.block_size)
