import torch
import torch.nn as nn
import torch.nn.functional as F
from models.operator.mask import Conv2d_mask, Linear_mask

import torch
from torch.nn import Conv2d, Linear, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math

initial_mask = 1
name = 'softplus'
thresh = 2

def f(x, name=name):
    if name == 'softplus':
        return F.softplus(x)
    elif name == 'sigmoid':
        return torch.sigmoid(x)
    elif name == 'exp':
        return torch.exp(x)
    elif name == 'cov':
        return 1 / 2 * (torch.tanh(x) + 1)
    elif name == 'identity':
        return x
    elif name == 'tanh':
        return 0.01 * torch.tanh(x)


def f_inv(x, name=name):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    if name == 'softplus':
        x = (0.0001) * (x == 0).float() + x * (x != 0).float()
        return torch.log(torch.exp(x) - 1)
    elif name == 'sigmoid':
        x = x * (x < 1).float() * (x > 0).float() + 0.999 * (x==1).float() + 0.001 * (x==0).float()
        return torch.log(x / (1-x))
    elif name == 'exp':
        x = 0.001 * (x == 0).float() + x * (x != 0).float()
        return torch.log(x)
    elif name == 'cov':
        return torch.atanh(2 * 0.99 * (x == 1).float() + 0.001 * (x == 0).float() + x * (0<x).float()*(x<1).float() - 1)
    elif name == 'identity':
        return x
    elif name == 'tanh':
        x = -0.999 * (x == -1).float() + 0.999 * (x == 1).float() + x * (-1 < x).float() * (x < 1).float()
        return torch.atanh(x / 0.01)


def operator(w, m):
    masked_weight = w * f(m)
    return masked_weight



def compute_prune_ratio(net, is_param=False):
    count = 0
    w_shape = 0
    for name, param in net.named_parameters():
        if not 'mask' in name:
            count += (param == 0).sum().item()
            w_shape += param.shape.numel()

    if is_param:
        return int(count) / w_shape, w_shape
    else:
        return int(count) / w_shape



def clamping_mask_network(modules, device):
    for m in modules:
        m.mask_weight.data.clamp_(min=f_inv(0).to(device), max=f_inv(1).to(device))
        if m.bias is not None:
            m.mask_bias.data.clamp_(min=f_inv(0).to(device), max=f_inv(1).to(device))

def reinitialize_mask_network(modules):
    for m in modules:
        m.mask_weight.data = f_inv(initial_mask) * torch.ones_like(m.mask_weight.data)
        if m.bias is not None:
            m.mask_bias.data = f_inv(initial_mask) * torch.ones_like(m.mask_bias.data)

def index2mask(index, modules, device):
    mask_dict = {}
    index = index.to(device)
    i=0
    for m in modules:
        mask_weight = torch.ones_like(m.mask_weight)
        j = i + mask_weight.shape[0]
        mask_weight = f_inv(index[i:j]).view(mask_weight.shape)
        i=j
        mask_dict[m] = [mask_weight]

    return mask_dict

class Conv2d_mask(Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 ):
        super(Conv2d_mask, self).__init__(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode)

        self.bias_bool = bias
        self.mask_weight = Parameter(f_inv(initial_mask) * torch.ones(self.weight.shape))
        self.mask_bias   = Parameter(f_inv(initial_mask)*torch.ones_like(self.bias)) if bias else None

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            operator(weight, self.mask_weight), operator(self.bias, self.mask_bias) if self.bias_bool else self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, operator(weight, self.mask_weight), operator(self.bias, self.mask_bias) if self.bias_bool else self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Linear_mask(Linear):
    def __init__(self, in_features, out_features, bias = True):
        super(Linear_mask, self).__init__(in_features, out_features, bias)
        self.mask_weight = Parameter(f_inv(initial_mask)*torch.ones(self.weight.shape))

        self.bias_bool = bias
        self.mask_bias = Parameter(f_inv(initial_mask)*torch.ones_like(self.bias)) if bias else None

    def forward(self, input):
        # Apply the mask during the forward pass
        masked_weight = operator(self.weight, self.mask_weight)
        masked_bias = operator(self.bias, self.mask_bias) if self.bias_bool else self.bias
        return F.linear(input, masked_weight, masked_bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_mask(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_mask(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_mask(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.conv1 = Conv2d_mask(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_mask(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_mask(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_mask(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mean=None, std=None, spatial_expansion=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)
        self.spatial_expansion = spatial_expansion

        self.conv1 = Conv2d_mask(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Linear_mask(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, pop=False, inter=False):
        # Feature visualization
        if not inter:
            out = (x - self.mean) / self.std
            out = F.relu(self.bn1(self.conv1(out)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            if pop:
                return out
        else:
            out = x

        if self.spatial_expansion:
            out = F.avg_pool2d(out, 8)
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet(depth=18, dataset='cifar10', mean=None, std=None):
    if dataset == 'cifar10' or dataset == 'svhn':
        num_classes = 10
        spatial_expansion = False
    elif dataset == 'cifar100':
        num_classes = 100
        spatial_expansion = False
    elif dataset == 'tiny':
        num_classes = 200
        spatial_expansion = True
    else:
        raise NotImplementedError


    if depth == 18:
        block = BasicBlock
        num_blocks = [2, 2, 2, 2]
    elif depth == 34:
        block = BasicBlock
        num_blocks = [3, 4, 6, 3]
    elif depth == 50:
        block = Bottleneck
        num_blocks = [3, 4, 6, 3]
    else:
        raise NotImplementedError

    return ResNet(block=block, num_blocks=num_blocks, num_classes=num_classes,
                  mean=mean, std=std, spatial_expansion=spatial_expansion)


_AFFINE = True
# _AFFINE = False

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, mean=None, std=None, init_weights=True, cfg=None):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

        self.feature = self.make_layers(cfg, True)
        self.dataset = dataset
        if dataset == 'cifar10' or dataset == 'svhn':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tiny':
            num_classes = 200
        self.classifier = Linear_mask(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d_mask(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=_AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, pop=False, inter=False):
        # Feature visualization
        if not inter:
            x = (x-self.mean) / self.std
            x = self.feature(x)
            if pop:
                return x


        if self.dataset == 'tiny':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d_mask):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
            elif isinstance(m, Linear_mask):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()