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


class GroupNormMoving(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5,
                momentum=1.0, affine=True,
                track_running_stats=True):
        super(GroupNormMoving, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps

        self.momentum = momentum
        self.affine = affine

        self.track_running_stats = track_running_stats

        tensor_shape = (1, num_features, 1, 1)

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(*tensor_shape))
            self.bias = nn.Parameter(torch.Tensor(*tensor_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.initial = True
        
        self.reset_parameters()

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0, "Channel must be divided by groups"

        x = x.view(N, G, -1)
        
        if self.training and self.track_running_stats:
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            if self.initial:
                self.running_mean.data = torch.zeros_like(mean)
                self.running_var.data =  torch.ones_like(var)
                self.initial = False
            
            with torch.no_grad():
                self.running_mean = mean * self.momentum + \
                                    self.running_mean * (1 - self.momentum)
                self.running_var = var * self.momentum + \
                                    self.running_var * (1 - self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

    def reset_parameters(self):
        if self.track_running_stats:
            if self.running_mean is not None and self.running_var is not None:
                self.running_mean.zero_()
                self.running_var.fill_(1)
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_groups}, {num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine}, track_running_stats={track_running_stats})'
                .format(name=self.__class__.__name__, **self.__dict__))
