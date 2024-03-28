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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


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
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, divided_by=1):
        super(ResNet, self).__init__()
        self.in_planes = 64 // divided_by

        self.conv1 = nn.Conv2d(3, 64 // divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64 // divided_by)
        self.layer1 = self._make_layer(block, 64 // divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128 // divided_by, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256 // divided_by, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512 // divided_by, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 // divided_by * block.expansion, num_classes)

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
        return out


class ResNet_adv(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, w=1):
        super(ResNet_adv, self).__init__()
        self.in_planes = int(4 * w)

        self.conv1 = nn.Conv2d(3, int(4 * w), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(4 * w))
        self.layer1 = self._make_layer(block, int(4 * w), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(8 * w), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(16 * w), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(32 * w), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(32 * w) * block.expansion, num_classes)

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
        return out


class ResNet_adv_wide(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, divided_by=1):
        super(ResNet_adv_wide, self).__init__()
        self.in_planes = 16 // divided_by

        self.conv1 = nn.Conv2d(3, 16 // divided_by, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 // divided_by)
        self.layer1 = self._make_layer(block, 160 // divided_by, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 320 // divided_by, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 640 // divided_by, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512//divided_by, num_blocks[3], stride=2)
        self.linear = nn.Linear(640 // divided_by * block.expansion, num_classes)

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
        #        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet18_adv(w=1):
    return ResNet_adv(BasicBlock, [2, 2, 2, 2], w=w)


def ResNet18_adv_wide():
    return ResNet_adv_wide(BasicBlock, [4, 3, 3], divided_by=1)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def ResNet18_1by16():
    return ResNet(BasicBlock, [2, 2, 2, 2], divided_by=16)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())



cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg16_1by8': [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64, 'M'],  # 1/8
    'vgg16_1by16': [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M'],  # 1/16
    'vgg16_1by32': [2, 2, 'M', 4, 4, 'M', 8, 8, 8, 'M', 16, 16, 16, 'M', 16, 16, 16, 'M']  # 1/32
}


class VGG(nn.Module):
    def __init__(self, vgg_name, w=16):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], w)
        final_channels = None
        if vgg_name == 'vgg16':
            final_channels = int(512 * w / 16)
        elif vgg_name == 'vgg16_1by8':
            final_channels = 64
        elif vgg_name == 'vgg16_1by16':
            final_channels = 32
        elif vgg_name == 'vgg16_1by32':
            final_channels = 16
        self.classifier = nn.Linear(final_channels, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, w):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(w / 16 * x)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_adv(nn.Module):
    def __init__(self, vgg_name, w=1):
        super(VGG_adv, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], w)

        final_channels = None
        self.base = 32
        # if vgg_name  == 'vgg16':
        final_channels = int(512 * w / 16)

        self.classifier = nn.Linear(final_channels, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, w):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(w / 16 * x)
                print('x is {}'.format(x))
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_ori_adv(nn.Module):
    def __init__(self, vgg_name, w=1):
        super(VGG_ori_adv, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], w)

        final_channels = None

        final_channels = int(512 * w / 16)

        self.classifier = nn.Linear(final_channels, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, w):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = int(w / 16 * x)
                print('x is {}'.format(x))
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),

                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()

