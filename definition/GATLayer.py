# -*- coding: utf-8 -*-
# @Time    : 2021/10/12 10:23
# @Author  : Naynix
# @File    : GATLayer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(input_dim, output_dim, bias=False)
        self.a1 = nn.Parameter(torch.zeros(size=(output_dim, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(output_dim, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        """
        只对输入进行维度转换，并计算其注意力系数，不进行特征融合
        :param input: batch * N * input_dim
        :param adj: batch * N * N
        :return:
        """
        h = self.W(input)
        batch_size, N, _ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        e = self.leakyrelu(middle_result1 + middle_result2)
        att = e.masked_fill(adj == 0, -1e9)
        att = F.softmax(att, dim=2)
        # att = F.dropout(att, self.dropout, training=self.training)

        return h, att

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, alpha, nheads, interval_num):
        super(GAT, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.interval_num = interval_num
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(self.input_dim, self.hidden_dim,
                                         dropout=self.dropout, alpha=alpha, concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, x, adj, graph_per_interval):
        """
        :param x: batch * N * input_dim
        :param adj: batch * N * N
        :param graph_per_interval: batch * interval_num * N * N
        :return:
        """
        _, N, _ = x.size() # N： 单样本中节点个数
        x = F.dropout(x, self.dropout, training=self.training)

        outputs = []
        for attention in self.attentions:
            h, att = attention(x, adj)
            output = h.repeat(1, self.interval_num, 1).view(-1, N, self.hidden_dim)
            att = att.repeat(1, self.interval_num, 1).view(-1, self.interval_num, N, N)
            att = torch.mul(att, graph_per_interval)
            att = att.masked_fill(att == 0, -1e9)
            att = F.softmax(att, dim=3)
            output = torch.bmm(att.view(-1, N, N), output).view(
                -1, self.interval_num, N, self.hidden_dim
            )
            output = F.elu(output)
            output = F.dropout(output, p=self.dropout)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-1)
        return outputs
