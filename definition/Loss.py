# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 20:03
# @Author  : Naynix
# @File    : Loss.py
from torch import nn
import torch
class Loss(nn.Module):
    def __ini__(self):
        pass

    def forward(self, outs, pred):
        batch_size = outs.shape[0]
        losses = torch.pow(((outs - pred) / pred), 2)
        loss = torch.sum(losses) / batch_size

        return loss