'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if isinstance(x, tuple):
            x, output_list = x
        else:
            output_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        # output_list.append(out)

        out = self.bn2(self.conv2(out))
        # output_list.append(out)

        out += self.shortcut(x)
        out = F.relu(out)
        output_list.append(out)

        return out, output_list


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if isinstance(x, tuple):
            x, output_list = x
        else:
            output_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        output_list.append(out)

        return out, output_list


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rob=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.rob = rob

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        output_list.append(out)

        out, out_list = self.layer1(out)
        output_list.extend(out_list)

        out, out_list = self.layer2(out)
        output_list.extend(out_list)

        out, out_list = self.layer3(out)
        output_list.extend(out_list)

        out, out_list = self.layer4(out)
        output_list.extend(out_list)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        output_list.append(out)

        out = self.linear(out)
        # out = F.log_softmax(self.linear(out), dim=1)
        # output_list.append(out)

        if self.rob:
            return out
        else:
            return out, output_list


def ResNet18(**kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    return ResNet(BasicBlock, [2, 2, 2, 2], kwargs['num_classes'], rob=True)


# def ResNet18(num_classes=10):
# return ResNet(BasicBlock, [2,2,2,2], num_classes=10, rob=False)

def ResNet34(**kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    return ResNet(BasicBlock, [3, 4, 6, 3], kwargs['num_classes'], rob=rob)


def ResNet50(**kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    return ResNet(Bottleneck, [3, 4, 6, 3], kwargs['num_classes'], rob=rob)


def ResNet101(**kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    return ResNet(Bottleneck, [3, 4, 23, 3], kwargs['num_classes'], rob=rob)


def ResNet152(**kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    return ResNet(Bottleneck, [3, 8, 36, 3], kwargs['num_classes'], rob=rob)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
}


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s, last=False):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    if last:
        layers += [nn.AdaptiveAvgPool2d((2, 2))]
    else:
        layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.ReLU()
    )
    return layer


class VGG16(nn.Module):
    def __init__(self, **kwargs):
        super(VGG16, self).__init__()

        self.rob = True  # kwargs['robustness'] if 'robustness' in kwargs else False
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2, last=True)

        # FC layers
        self.layer6 = vgg_fc_layer(2048, 256)
        self.layer7 = vgg_fc_layer(256, 256)

        # Final layer
        self.layer8 = nn.Linear(256, 10)

    def forward(self, x):
        output_list = []

        out = self.layer1(x)
        output_list.append(out)

        out = self.layer2(out)
        output_list.append(out)

        out = self.layer3(out)
        output_list.append(out)

        out = self.layer4(out)
        output_list.append(out)

        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)

        # print(out.shape)
        out = self.layer6(out)
        output_list.append(out)

        out = self.layer7(out)
        output_list.append(out)

        out = self.layer8(out)

        if self.rob:
            return out
        else:
            return out, output_list