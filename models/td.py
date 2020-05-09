import torch
import torch.nn as nn
import torch.nn.functional as F


class Greater_Than(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.gt(input, 0).float()

    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class Conv2d_TD(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, gamma=0.0, alpha=0.0, block_size=16,
                 cg_groups=1, cg_threshold_init=-6.0, cg_alpha=2.0, cg_grouping=True):
        super(Conv2d_TD, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.gamma = gamma
        self.alpha = alpha
        self.block_size = block_size
        self.cg_groups = cg_groups
        if self.cg_groups > 1:
            self.cg_alpha=cg_alpha
            self.cg_threshold = nn.Parameter(cg_threshold_init * torch.ones(1, out_channels, 1, 1))
            self.cg_in_chunk_size = int(in_channels / self.cg_groups)
            #self.cg_bn = nn.BatchNorm2d(out_channels, affine=False)
            self.cg_bn = nn.functional.instance_norm
            self.cg_gt = Greater_Than.apply
            self.cg_grouping = cg_grouping
            self.mask_base = torch.zeros_like(self.weight.data).cuda()
            if self.cg_grouping:
                self.cg_out_chunk_size = int(out_channels / self.cg_groups)                 
                for i in range(self.cg_groups):
                    self.mask_base[i*self.cg_out_chunk_size:(i+1)*self.cg_out_chunk_size,i*self.cg_in_chunk_size:(i+1)*self.cg_in_chunk_size,:,:] = 1
            else:
                self.mask_base[:,self.cg_base_group_index*self.cg_in_chunk_size:(self.cg_base_group_index+1)*self.cg_in_chunk_size,:,:] = 1
            self.mask_cond = 1 - self.mask_base

            self.cg_base_group_index = 0
            #self.cg_relu = lambda x: x
    
    def forward(self, input):
        if self.gamma > 0 and self.alpha > 0:
            if self.cg_groups > 1:
                # get the partial sum from base path
                self.Yp = F.conv2d(input, self.weight * self.mask_base * self.mask_keep_original, None,
                        self.stride, self.padding, self.dilation, self.groups)
                # identify important regions
                self.d = self.cg_gt(torch.sigmoid(self.cg_alpha*(self.cg_bn(self.Yp)-self.cg_threshold))-0.5)
                #self.d = self.cg_gt(self.cg_bn(self.Yp)-self.cg_threshold)
                # report statistics
                self.num_out = self.d.numel()
                self.num_full = self.d[self.d>0].numel()
                self.Yc = F.conv2d(input, self.weight * self.mask_cond * self.mask_keep_original, None, 
                        self.stride, self.padding, self.dilation, self.groups)
                out = self.Yp + self.Yc * self.d
            else:
                out = F.conv2d(input, self.weight * self.mask_keep_original, None, self.stride, self.padding,
                                        self.dilation, self.groups)
        else:
            if self.cg_groups > 1:
                # get the partial sum from base path
                self.Yp = F.conv2d(input, self.weight * self.mask_base, None,
                        self.stride, self.padding, self.dilation, self.groups)
                # identify important regions
                self.d = self.cg_gt(torch.sigmoid(self.cg_alpha*(self.cg_bn(self.Yp)-self.cg_threshold))-0.5)
                #self.d = self.cg_gt(self.cg_bn(self.Yp)-self.cg_threshold)
                # report statistics
                self.num_out = self.d.numel()
                self.num_full = self.d[self.d>0].numel()
                self.Yc = F.conv2d(input, self.weight * self.mask_cond, None, 
                        self.stride, self.padding, self.dilation, self.groups)
                out = self.Yp + self.Yc  * self.d
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

    def forward(self, input):
        if self.gamma > 0 and self.alpha > 0:
            return F.linear(input, self.weight * self.mask_keep_original, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return super(Linear_TD, self).extra_repr() + ', gamma={}, alpha={}, block_size={}'.format(
                self.gamma, self.alpha, self.block_size)
