import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def rate_act_func(k_score, k_min):
    k = torch.sigmoid(k_score)
    k = k * (1 - k_min)  # E.g. global_k = 0.1, Make layer k in range [0.0, 0.99]
    k = k + k_min  # Make layer k in range [0.01, 1.0]
    return k
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, prune_reg='weight'):
        # Get the subnetwork by sorting the scores and using the top k%

        if prune_reg == 'weight':
            # """ Weight pruning
            out = scores.clone()
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())

            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            # """

        elif prune_reg == 'channel':
            out = scores.clone()
            kept_weights = torch.topk(out, k=int(torch.round(k*out.shape[1])), dim=1).indices
            out = torch.transpose(out, 0,1)
            out[:] = 0
            out[kept_weights] = 1
            out = torch.transpose(out, 0,1)
            # """

        else:
            raise NameError('Please check prune_reg, current "{}" is not in [weight, channel] !'.format(prune_reg))

        ctx.save_for_backward(out)

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.

        g_r = torch.sum(g)

        return g, g_r, None


class SubnetConv(nn.Conv2d):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            prune_reg='weight',
            task_mode='harp_prune'
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.prune_reg = prune_reg
        self.task_mode = task_mode

        if self.prune_reg == 'weight':
            # Weight pruning or Filter Pruning
            self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        elif self.prune_reg == 'channel' and task_mode in ['score_finetune', 'rate_finetune', 'harp_finetune']:
            # Channel Finetuning or Resume Pruning
            self.popup_scores = Parameter(torch.Tensor(torch.Size([1,self.weight.shape[1],1,1])))
        elif self.prune_reg == 'channel' and task_mode in ['score_prune', 'rate_prune', 'harp_prune']:
            # Channel Pruning
            self.popup_scores = Parameter(torch.Tensor(torch.Size([self.weight.shape[0], 1,1,1])))
        else:
            raise NameError('prune_reg "{}" or task_mode "{}" are not correct!'.format(prune_reg, task_mode))

        self.k_score = Parameter(torch.Tensor(torch.Size([])))

        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        nn.init.constant_(self.k_score, 1.0)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0

    def set_prune_rate(self, k, global_k, alpha, device):
        self.k = k if self.task_mode != 'pretrain' else 1.0
        self.k_min = global_k * alpha if self.task_mode != 'pretrain' else 0.0

    def forward(self, x):

        if self.task_mode == 'pretrain':
            k = 1.0
        else:
            k = rate_act_func(self.k_score, self.k_min)

        adj = GetSubnet.apply(self.popup_scores.abs(), k, self.prune_reg)

        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.

    def __init__(self, in_features, out_features, bias=True, prune_reg='weight', task_mode='harp_prune'):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)

        self.prune_reg = prune_reg
        self.task_mode = task_mode

        if self.prune_reg == 'weight':
            # Weight pruning or Filter Pruning
            self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        elif self.prune_reg == 'channel' and task_mode in ['score_finetune', 'rate_finetune', 'harp_finetune']:
            # Channel Finetuning or Resume Pruning
            self.popup_scores = Parameter(torch.Tensor(torch.Size([1,self.weight.shape[1]])))
        elif self.prune_reg == 'channel' and task_mode in ['score_prune', 'rate_prune', 'harp_prune']:
            # Channel Pruning
            self.popup_scores = Parameter(torch.Tensor(torch.Size([self.weight.shape[0],1])))
        else:
            raise NameError('prune_reg "{}" or task_mode "{}" are not correct!'.format(prune_reg, task_mode))

        # self.k_score = Parameter(torch.Tensor([0]))
        self.k_score = Parameter(torch.Tensor(torch.Size([])))

        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        nn.init.constant_(self.k_score, 1.0)
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.w = 0
        # self.register_buffer('w', None)

    def set_prune_rate(self, k, global_k, alpha, device):
        self.k = k if self.task_mode != 'pretrain' else 1.0
        self.k_min = global_k * alpha if self.task_mode != 'pretrain' else 0.0

    def forward(self, x):

        if self.task_mode == 'pretrain':
            k = 1.0
        else:
            k = rate_act_func(self.k_score, self.k_min)

        adj = GetSubnet.apply(self.popup_scores.abs(), k, self.prune_reg)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)

        return x