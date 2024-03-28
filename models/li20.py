'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pruning.utils import to_var
from torch.autograd import Variable


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

class MaskedBN2d(nn.BatchNorm2d):
    def __init__(self, out_channels):
        super(MaskedBN2d, self).__init__(out_channels)
        self.out_channels = out_channels
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.bias.data = self.bias.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training)


class MaskedBN1d(nn.BatchNorm1d):
    def __init__(self, out_channels):
        super(MaskedBN1d, self).__init__(out_channels)
        self.out_channels = out_channels
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.bias.data = self.bias.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.bias.data = self.bias.data * self.mask.data.mean(1)
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
        self.bias_flag = bias
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        if self.bias_flag:
            self.bias.data.zero_()

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        if self.bias_flag:
            self.bias.data = self.bias.data * self.mask.data.sum((1, 2, 3)) / (
                        self.mask.shape[1] * self.mask.shape[2] * self.mask.shape[3])
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MaskedBN2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MaskedBN2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                MaskedBN2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = MaskedBN2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MaskedBN2d(planes)
        self.conv3 = MaskedConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = MaskedBN2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                MaskedBN2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = MaskedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MaskedBN2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = MaskedLinear(512 * block.expansion, num_classes)

        self.index = 0
        for m in self.modules():
            # print(m.__class__.__name__)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                self.index += 1
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.index += 1
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                self.index += 1
            else:
                pass
        # print("index init:", self.index)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = self.model(x)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2])
        self.index = self.model.index

    def forward(self, x):
        out = self.model(x)
        return out

    def set_masks(self, masks, transfer=False):

        index = np.array([0])

        for layer in self.model.modules():
            # print(m.__class__.__name__)
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                # print(layer.__class__.__name__)
                layer.set_mask(torch.from_numpy(masks[index[0]]))
                index[0] += 1

        assert index[0] == self.index, "seif.index[" + str(self.index) + "] !=" \
                                                                         "mask index[" + str(index[0]) + "]"


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = ResNet(BasicBlock, [3, 4, 6, 3])
        self.index = self.model.index

    def forward(self, x):
        out = self.model(x)
        return out

    def set_masks(self, masks, transfer=False):

        index = np.array([0])

        for layer in self.model.modules():
            # print(m.__class__.__name__)
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                # print(layer.__class__.__name__)
                layer.set_mask(torch.from_numpy(masks[index[0]]))
                index[0] += 1

        assert index[0] == self.index, "seif.index[" + str(self.index) + "] !=" \
                                                                         "mask index[" + str(index[0]) + "]"


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


defaultcfg = {
    7: [64, 'M', 128, 'M', 256, 'M', 512],
    '8s': [32, 'M', 32, 'M', 64, 'M', 64, 'M', 64],
    '8m': [32, 'M', 32, 'M', 64, 'M', 128, 'M', 128],
    8: [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512],
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class vgg(nn.Module):
    # ref: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/l1-norm-pruning/models/vgg.py
    def __init__(self, dataset='cifar', depth=16, init_weights=True, cfg=None, cap_ratio=1.0):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg
        self.depth = depth
        if depth == "8s": self.depth = 8
        self.cap_ratio = cap_ratio
        self.index = 0

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar' or dataset == 'svhn' or dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(
            MaskedLinear(math.ceil(cfg[-1] * cap_ratio), math.ceil(512 * cap_ratio)),
            MaskedBN1d(math.ceil(512 * cap_ratio)),
            nn.ReLU(inplace=True),
            MaskedLinear(math.ceil(512 * cap_ratio), num_classes)
        )
        self.model = nn.Sequential(
            self.feature, nn.AvgPool2d(2), Flatten(), self.classifier)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = math.ceil(v * self.cap_ratio)
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, MaskedBN2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        '''
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.model(x)
        '''
        return y

    def _initialize_weights(self):
        for m in self.model.modules():
            # print(self.index, m.__class__.__name__)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # print(n, m.weight.data.normal_(0, math.sqrt(2. / n))[0])
                if m.bias is not None:
                    m.bias.data.zero_()
                self.index += 1
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
                self.index += 1
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                self.index += 1
            elif isinstance(m, nn.BatchNorm1d):
                self.index += 1

    def set_masks(self, masks, transfer=False):
        def set_layer_mask(i, masks, index):
            if isinstance(i, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                # print(index[0], i.__class__.__name__)
                if transfer and index[0] >= (masks.shape[0] - 1):
                    # print(i.weight.shape, masks[index[0]].shape)
                    pass
                else:
                    # print(masks[index[0]].shape)
                    i.set_mask(torch.from_numpy(masks[index[0]]))
                    # print(i.weight.shape)
                index[0] += 1

        index = np.array([0])
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for i in layer.children():
                    set_layer_mask(i, masks, index)
            else:
                set_layer_mask(layer, masks, index)
        assert index[
                   0] == 2 * self.depth - 3 == self.index  # Again a hack to make sure that masks are provided for each layer.

    def transfer_model(self, net_orig):
        index = 0
        orig_index = 0
        orig_layers = []

        for layer in net_orig.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                # print(layer.__class__.__name__)
                orig_layers.append(layer)
                orig_index += 1

        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                # print(layer.__class__.__name__)

                if index >= len(orig_layers) - 1:
                    # print("last", index, layer.weight.data.shape, orig_layers[index].weight.data.shape)
                    layer.weight.data = orig_layers[index].weight.data.clone().repeat(10, 1).detach()

                if index < len(orig_layers) - 1:
                    layer.weight.data = orig_layers[index].weight.data.clone().detach()
                    # print(index, layer.weight.data.shape, orig_layers[index].weight.data.shape)
                    index += 1


class vgg_no_bn(nn.Module):
    # ref: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/l1-norm-pruning/models/vgg.py
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None, cap_ratio=1.0):
        super(vgg_no_bn, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg
        self.depth = depth
        if depth == "8s": self.depth = 8
        self.cap_ratio = cap_ratio

        self.feature = self.make_layers(cfg, False)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(
            MaskedLinear(math.ceil(cfg[-1] * cap_ratio), math.ceil(512 * cap_ratio)),
            # MaskedBN1d(math.ceil(512*cap_ratio)),
            nn.ReLU(inplace=True),
            MaskedLinear(math.ceil(512 * cap_ratio), num_classes)
        )
        self.model = nn.Sequential(
            self.feature, nn.AvgPool2d(2), Flatten(), self.classifier)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = math.ceil(v * self.cap_ratio)
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, MaskedBN2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        '''
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.model(x)
        '''
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_masks(self, masks):
        def set_layer_mask(i, masks, index):
            if isinstance(i, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                i.set_mask(torch.from_numpy(masks[index[0]]))
                index[0] += 1

        index = np.array([0])
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for i in layer.children():
                    set_layer_mask(i, masks, index)
            else:
                set_layer_mask(layer, masks, index)
        # assert index[0] == 2*self.depth - 3 #Again a hack to make sure that masks are provided for each layer.

