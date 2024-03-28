# resnet18
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, conv_layer, stride=1, groups=1, dilation=1, prune_reg='weight', task_mode='harp_prune'):
    """3x3 convolution with padding"""
    return conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, prune_reg=prune_reg, task_mode=task_mode)


def conv1x1(in_planes, out_planes, conv_layer, stride=1, prune_reg='weight', task_mode='harp_prune'):
    """1x1 convolution"""
    return conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, dilation=1, prune_reg=prune_reg, task_mode=task_mode)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, conv_layer, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, prune_reg='weight', task_mode='harp_prune'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, conv_layer, stride, prune_reg=prune_reg, task_mode=task_mode)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_layer, prune_reg=prune_reg, task_mode=task_mode)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, conv_layer, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, prune_reg='weight', task_mode='harp_prune'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, conv_layer, prune_reg=prune_reg, task_mode=task_mode)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, conv_layer, stride, groups, dilation, prune_reg=prune_reg, task_mode=task_mode)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, conv_layer, prune_reg=prune_reg, task_mode=task_mode)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        # print(out.shape)
        # print(out[0][0])
        # print(out.sum())

        return out


class ResNet(nn.Module):

    def __init__(self, conv_layer, linear_layer, block, layers, mean, std,
                 num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, prune_reg='weight',
                 task_mode='harp_prune', normalize=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv_layer = conv_layer

        self.normalize = normalize
        self.mean = torch.Tensor(mean).unsqueeze(1).unsqueeze(1)
        self.std = torch.Tensor(std).unsqueeze(1).unsqueeze(1)
        self.num_classes = num_classes

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = conv_layer(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, prune_reg=prune_reg, task_mode=task_mode)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], prune_reg=prune_reg, task_mode=task_mode)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], prune_reg=prune_reg, task_mode=task_mode)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], prune_reg=prune_reg, task_mode=task_mode)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], prune_reg=prune_reg, task_mode=task_mode)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = linear_layer(512 * block.expansion, num_classes, prune_reg=prune_reg, task_mode=task_mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, prune_reg='weight', task_mode='harp_prune'):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, self.conv_layer, stride,
                        prune_reg=prune_reg, task_mode=task_mode),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.conv_layer, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, prune_reg=prune_reg, task_mode=task_mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.conv_layer, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, prune_reg=prune_reg, task_mode=task_mode))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# NOTE: Only supporting default (kaiming_init) initializaition.
def ResNet18(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for Resnets"
    return ResNet(conv_layer, linear_layer, BasicBlock, [2, 2, 2, 2], **kwargs)


import torch
import torch.nn as nn
import math


class VGG(nn.Module):
    def __init__(self, features, last_conv_channels_len, linear_layer, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],
                 num_classes=10, prune_reg='weight', task_mode='prune', normalize=False):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.normalize = normalize
        self.mean = torch.Tensor(mean).unsqueeze(1).unsqueeze(1)
        self.std = torch.Tensor(std).unsqueeze(1).unsqueeze(1)
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            linear_layer(last_conv_channels_len * 2 * 2, 256, prune_reg=prune_reg, task_mode=task_mode),
            nn.ReLU(True),
            linear_layer(256, 256, prune_reg=prune_reg, task_mode=task_mode),
            nn.ReLU(True),
            linear_layer(256, num_classes, prune_reg=prune_reg, task_mode=task_mode),
        )

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def initialize_weights(model, init_type):
    print(f"Initializing model with {init_type}")
    assert init_type in ["kaiming_normal", "kaiming_uniform", "signed_const"]
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if init_type == "signed_const":
                n = math.sqrt(
                    2.0 / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels)
                )
                m.weight.data = m.weight.data.sign() * n
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
            if init_type == "signed_const":
                n = math.sqrt(2.0 / m.in_features)
                m.weight.data = m.weight.data.sign() * n
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def make_layers(cfg, conv_layer, mean=None, std=None, normalize=None, batch_norm=True, num_classes=10, prune_reg='weight', task_mode='prune'):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1, bias=False, prune_reg=prune_reg, task_mode=task_mode)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "2": [64, "M", 64, "M"],
    "4": [64, 64, "M", 128, 128, "M"],
    "6": [64, 64, "M", 128, 128, "M", 256, 256, "M"],
    "8": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M"],
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
}


def vgg16(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["16"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["16"], conv_layer, batch_norm=False, **kwargs), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg16_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["16"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["16"], conv_layer, batch_norm=True, **kwargs), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model

