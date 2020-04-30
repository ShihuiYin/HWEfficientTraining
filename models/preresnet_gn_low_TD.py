
"""
    PreResNet model definition
    ported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
"""

import torch.nn as nn
import math
from qtorch import FloatingPoint
from qtorch.quant import Quantizer
from .td import Conv2d_TD, Linear_TD

__all__ = ['PreResNet110LP_TD_GN', 'PreResNet20LP_TD_GN']


def conv3x3(in_planes, out_planes, stride=1, gamma=0.0, alpha=0.0, block_size=1):
    return Conv2d_TD(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, gamma=gamma, alpha=alpha, block_size=block_size)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, quant, stride=1, downsample=None, gamma=0.0, alpha=0.0, block_size=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.GroupNorm(4, inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride, gamma=gamma, alpha=alpha, block_size=block_size)
        self.bn2 = nn.GroupNorm(4, planes)
        self.conv2 = conv3x3(planes, planes, gamma=gamma, alpha=alpha, block_size=block_size)
        self.downsample = downsample
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv1(out)
        out = self.quant(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, quant, stride=1, downsample=None, gamma=0.0, alpha=0.0, block_size=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = Conv2d_TD(inplanes, planes, kernel_size=1, bias=False,
                        gamma=gamma, alpha=alpha, block_size=block_size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_TD(planes, planes, kernel_size=3, stride=stride,
                        padding=1, bias=False, gamma=gamma, alpha=alpha, block_size=block_size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_TD(planes, planes * 4, kernel_size=1, bias=False, 
                        gamma=gamma, alpha=alpha, block_size=block_size)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv1(out)
        out = self.quant(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv3(out)
        out = self.quant(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):

    def __init__(self,quant, num_classes=10, depth=110, gamma=0.0, alpha=0.0, block_size=1, cg_groups=1, cg_threshold_init=0, cg_alpha=2):

        super(PreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, quant, gamma=gamma, alpha=alpha, block_size=block_size)
        self.layer2 = self._make_layer(block, 32, n, quant, stride=2, gamma=gamma, alpha=alpha, block_size=block_size)
        self.layer3 = self._make_layer(block, 64, n, quant, stride=2, gamma=gamma, alpha=alpha, block_size=block_size)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.quant = quant()
        IBM_half = FloatingPoint(exp=6, man=9)
        self.quant_half = Quantizer(IBM_half, IBM_half, "nearest", "nearest")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, quant, stride=1, gamma=0.0, alpha=0.0, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d_TD(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, gamma=gamma, alpha=alpha, block_size=block_size),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, quant , stride, downsample, gamma=gamma, alpha=alpha, block_size=block_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, quant, gamma=gamma, alpha=alpha, block_size=block_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_half(x)
        x = self.conv1(x)
        x = self.quant(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)
        x = self.quant(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.quant_half(x)

        return x


class PreResNet110LP_TD_GN:
    base = PreResNet
    args = list()
    kwargs = {'depth': 110}


class PreResNet20LP_TD_GN:
    base = PreResNet
    args = list()
    kwargs = {'depth': 20}
