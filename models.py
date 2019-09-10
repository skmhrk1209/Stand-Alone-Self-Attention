import torch
from torch import nn
from modules import *
from collections import *


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, expansion=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.actv1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.actv2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.actv3 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, bias=False)
        self.projection = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, inputs):
        outputs = self.norm1(inputs)
        outputs = self.actv1(outputs)
        shortcut = self.projection(outputs)
        outputs = self.conv1(outputs)
        outputs = self.norm2(outputs)
        outputs = self.actv2(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm3(outputs)
        outputs = self.actv3(outputs)
        outputs = self.conv3(outputs)
        outputs += shortcut
        return outputs


class SelfAttentionResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, expansion=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.actv1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.actv2 = nn.ReLU()
        self.conv2 = SelfAttention(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.actv3 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, bias=False)
        self.projection = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, inputs):
        outputs = self.norm1(inputs)
        outputs = self.actv1(outputs)
        shortcut = self.projection(outputs)
        outputs = self.conv1(outputs)
        outputs = self.norm2(outputs)
        outputs = self.actv2(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm3(outputs)
        outputs = self.actv3(outputs)
        outputs = self.conv3(outputs)
        outputs += shortcut
        return outputs
        

class ResNet(nn.Module):

    def __init__(self, conv_param, pool_param, residual_params, num_classes):

        super().__init__()

        self.network = nn.Sequential(OrderedDict(
            first_conv_block=nn.Sequential(OrderedDict(
                conv=nn.Conv2d(**conv_param),
                pool=nn.MaxPool2d(**pool_param)
            )),
            residual_blocks=nn.Sequential(*[
                nn.Sequential(
                    nn.Sequential(*[
                        ResidualBlock(
                            in_channels=residual_param.in_channels,
                            out_channels=residual_param.out_channels,
                            kernel_size=residual_param.kernel_size,
                            padding=residual_param.padding,
                            stride=residual_param.stride,
                            expansion=residual_param.expansion
                        ) for _ in range(residual_param.blocks)[:1]
                    ]),
                    nn.Sequential(*[
                        ResidualBlock(
                            in_channels=residual_param.out_channels * residual_param.expansion,
                            out_channels=residual_param.out_channels,
                            kernel_size=residual_param.kernel_size,
                            padding=residual_param.padding,
                            expansion=residual_param.expansion
                        ) for _ in range(residual_param.blocks)[1:]
                    ])
                ) for residual_param in residual_params
            ]),
            last_conv_block=nn.Sequential(OrderedDict(
                norm=nn.BatchNorm2d(residual_params[-1].out_channels * residual_params[-1].expansion),
                actv=nn.ReLU(),
                pool=nn.AdaptiveAvgPool2d(1),
                conv=nn.Conv2d(residual_params[-1].out_channels * residual_params[-1].expansion, num_classes, 1)
            ))
        ))

    def forward(self, inputs):
        return self.network(inputs)


class SelfAttentionResNet(nn.Module):

    def __init__(self, conv_param, pool_param, residual_params, num_classes):

        super().__init__()

        self.network = nn.Sequential(OrderedDict(
            first_conv_block=nn.Sequential(OrderedDict(
                conv=AttentionStem(**conv_param),
                pool=nn.MaxPool2d(**pool_param)
            )),
            residual_blocks=nn.Sequential(*[
                nn.Sequential(
                    nn.Sequential(*[
                        SelfAttentionResidualBlock(
                            in_channels=residual_param.in_channels,
                            out_channels=residual_param.out_channels,
                            kernel_size=residual_param.kernel_size,
                            padding=residual_param.padding,
                            stride=residual_param.stride,
                            expansion=residual_param.expansion
                        ) for _ in range(residual_param.blocks)[:1]
                    ]),
                    nn.Sequential(*[
                        SelfAttentionResidualBlock(
                            in_channels=residual_param.out_channels * residual_param.expansion,
                            out_channels=residual_param.out_channels,
                            kernel_size=residual_param.kernel_size,
                            padding=residual_param.padding,
                            expansion=residual_param.expansion
                        ) for _ in range(residual_param.blocks)[1:]
                    ])
                ) for residual_param in residual_params
            ]),
            last_conv_block=nn.Sequential(OrderedDict(
                norm=nn.BatchNorm2d(residual_params[-1].out_channels * residual_params[-1].expansion),
                actv=nn.ReLU(),
                pool=nn.AdaptiveAvgPool2d(1),
                conv=nn.Conv2d(residual_params[-1].out_channels * residual_params[-1].expansion, num_classes, 1)
            ))
        ))

    def forward(self, inputs):
        return self.network(inputs)


