"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms
from qtorch import FloatingPoint
from qtorch.quant import Quantizer
from .td import Conv2d_TD, Linear_TD

__all__ = ['VGG16LP_TD', 'VGG16BNLP_TD', 'VGG16GNLP_TD', 'VGG19LP_TD', 'VGG19BNLP_TD', 'VGG19GNLP_TD']


def make_layers(cfg, quant, batch_norm=False, group_norm=False, gamma=0.5, alpha=0.5, block_size=16):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            use_quant = v[-1] != 'N'
            filters = int(v) if use_quant else int(v[:-1])
            if in_channels == 3: # or gamma == 0 or alpha == 0:
                conv2d = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            else:
                conv2d = Conv2d_TD(in_channels, filters, kernel_size=3, padding=1, gamma=gamma, 
                                    alpha=alpha, block_size=block_size)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            elif group_norm:
                layers += [conv2d, nn.GroupNorm(int(filters/16), filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            if use_quant: layers += [quant()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)


cfg = {
    16: ['64', '64', 'M', '128', '128', 'M', '256', '256', '256', 'M', '512', '512', '512', 'M', '512', '512', '512', 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, quant=None, num_classes=10, depth=16, batch_norm=False, group_norm=False, gamma=0.5, alpha=0.5, block_size=16):

        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], quant, batch_norm, group_norm, gamma, alpha, block_size)
        IBM_half = FloatingPoint(exp=6, man=9)
        quant_half = lambda : Quantizer(IBM_half, IBM_half, "nearest", "nearest")
        self.classifier = nn.Sequential(
            quant_half(),
            nn.Dropout(),
            Linear_TD(512, 512, gamma=gamma, alpha=alpha, block_size=block_size),
            nn.ReLU(True),
            quant(),
            nn.Dropout(),
            Linear_TD(512, 512, gamma=gamma, alpha=alpha, block_size=block_size),
            nn.ReLU(True),
            quant(),
            nn.Linear(512, num_classes),
            quant_half()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class VGG16LP_TD(Base):
    pass


class VGG16BNLP_TD(Base):
    kwargs = {'batch_norm': True}

class VGG16GNLP_TD(Base):
    kwargs = {'group_norm': True}

class VGG19LP_TD(Base):
    kwargs = {'depth': 19}


class VGG19BNLP_TD(Base):
    kwargs = {'depth': 19, 'batch_norm': True}

class VGG19GNLP_TD(Base):
    kwargs = {'depth': 19, 'group_norm': True}
