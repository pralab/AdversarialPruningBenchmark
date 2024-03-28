import torch
import torch.nn as nn
import math


class VGG(nn.Module):
    def __init__(self, features, last_conv_channels_len, linear_layer, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            linear_layer(last_conv_channels_len * 2 * 2, 256),
            nn.ReLU(True),
            linear_layer(256, 256),
            nn.ReLU(True),
            linear_layer(256, num_classes),
        )

    def forward(self, x):
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


def make_layers(cfg, conv_layer, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = conv_layer(in_channels, v, kernel_size=3, padding=1, bias=False)
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
        make_layers(cfgs["16"], conv_layer, batch_norm=False), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


def vgg16_bn(conv_layer, linear_layer, init_type, **kwargs):
    n = [i for i in cfgs["16"] if isinstance(i, int)][-1]
    model = VGG(
        make_layers(cfgs["16"], conv_layer, batch_norm=True), n, linear_layer, **kwargs
    )
    initialize_weights(model, init_type)
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_layer = conv_layer

        self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = linear_layer(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
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
        return out


# NOTE: Only supporting default (kaiming_init) initializaition.
def resnet18(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for Resnets"
    return ResNet(conv_layer, linear_layer, BasicBlock, [2, 2, 2, 2], **kwargs)
