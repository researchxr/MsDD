# -*- coding: utf-8 -*-
# @Time    : 2021/10/12 10:21
# @Author  : Naynix
# @File    : MD_Dynamic_net.py
import torch
from torch.nn import functional as F
import torch.nn as nn
import math
from definition.GATLayer import GAT
from definition.WordEncoder import WordEncoder,AttAggregate
from definition.TimeEncoder import TimeEncoding
from dgl.nn.pytorch.conv import GATConv, GraphConv
import dgl


class MD_Dynamic_net(nn.Module):
    def __init__(self, input_dim, user_dim, hidden_dim, output_dim, seq_len, post_num, interval_num,
                 lstm_layer_num, gat_layer_num, nheads, dropout, alpha, pool):
        super(MD_Dynamic_net, self).__init__()
        self.input_dim = input_dim
        self.user_dim = user_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.post_num = post_num
        self.interval_num = interval_num
        self.lstm_layer_num = lstm_layer_num
        self.gat_layer_num = gat_layer_num
        self.nheads = nheads
        self.dropout = dropout
        self.alpha = alpha
        self.pool = pool

        self.word_encoder = WordEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                                        lstm_layer_num=self.lstm_layer_num, dropout=self.dropout)


        self.bn = nn.BatchNorm1d(self.post_num)
        # self.t_bn = nn.BatchNorm1d(self.interval_num)
        # self.st_bn = nn.BatchNorm1d(self.interval_num)

        self.t_hidden_dim = self.hidden_dim
        self.t_lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim // 2,
                              num_layers=self.lstm_layer_num, batch_first=True, bidirectional=True)
        self.t_att_post = AttAggregate(input_dim=self.t_hidden_dim, dropout=self.dropout)

        self.st_hidden_dim = self.hidden_dim * self.nheads
        self.gat = GAT(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                       alpha=self.alpha, nheads=self.nheads, interval_num=self.interval_num)
        self.st_att_post = AttAggregate(input_dim=self.hidden_dim*self.nheads, dropout=self.dropout)
        
        self.lstm_post = nn.LSTM(input_size=self.t_hidden_dim+self.st_hidden_dim,
                                 hidden_size=self.t_hidden_dim+self.st_hidden_dim,
                                 num_layers=self.lstm_layer_num, batch_first=True, bidirectional=True)

        if self.pool == "mean":
            self.pool_graph = nn.AvgPool2d(kernel_size=(self.interval_num, 1), stride=1)
        else:
            self.pool_graph = nn.MaxPool2d(kernel_size=(self.interval_num, 1), stride=1)

        self.mlp = nn.Sequential(
            nn.Linear((self.t_hidden_dim + self.st_hidden_dim) * 2, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 2)
        )
        # self.reset_para()

    def reset_para(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.414)

    def forward(self, content_embeddings, user_embeddings, adj,
                graph_per_interval, content_mask, post_masks):
        """
        :param content_embeddings: batch * post_num * seq_len * input_dim 文本内容表示
        :param user_embeddings: batch * post_num * user_dim 用户特征表示
        :param adj: batch * post_num * post_num 节点关系矩阵
        :param graph_per_interval: batch * interval_num * post_num * post_num 各时间段的关系矩阵
        :param content_mask: batch * post_num * seq_len 内容掩码（标记token是否存在）
        :param post_masks: batch * interval_num * post_num 节点掩码（标记节点是否存在）
        :return:
        """
                    

        # Single-stage Embedding
        #the initial information
        content_embeddings = content_embeddings.view(-1, self.seq_len, self.input_dim)
        content_mask = content_mask.view(-1, self.seq_len)
        content_embeddings, word_score = self.word_encoder(content_embeddings,
                                               content_mask)
        content_embeddings = content_embeddings.view(-1, self.post_num, self.hidden_dim)

        content_embeddings = self.bn(content_embeddings)

        #Sequence Attention Module
        t_content_embeddings, _ = self.t_lstm(content_embeddings)
        t_content_embeddings = t_content_embeddings.repeat(1, self.interval_num, 1)
        t_content_embeddings = t_content_embeddings.view(-1, self.post_num, self.t_hidden_dim)
        post_masks = post_masks.view(-1, self.post_num)
        t_content_embeddings, t_post_score = self.t_att_post(t_content_embeddings, post_masks)
        t_content_embeddings = t_content_embeddings.view(-1, self.interval_num, self.t_hidden_dim)
        # t_content_embeddings = self.t_bn(t_content_embeddings)

        # Graph Attention Module
        st_content_embeddings = self.gat(content_embeddings, adj, graph_per_interval)
        repr = content_embeddings

        # 
        st_content_embeddings = st_content_embeddings.view(-1, self.post_num, self.hidden_dim * self.nheads)
        post_masks = post_masks.view(-1, self.post_num)
        st_content_embeddings, st_post_score = self.st_att_post(st_content_embeddings, post_masks)
        st_content_embeddings = st_content_embeddings.view(-1, self.interval_num, self.hidden_dim * self.nheads)
        # st_content_embeddings = self.st_bn(st_content_embeddings)

        # Multi-stage Dynamic Learning
        content_embeddings = torch.cat([st_content_embeddings, t_content_embeddings], dim=-1)
        content_embeddings_T = content_embeddings.permute(0, 2, 1)
        distance = torch.bmm(content_embeddings, content_embeddings_T)
        mask = torch.ones((self.interval_num, self.interval_num), dtype=torch.float).cuda()
        mask = torch.triu(mask, diagonal=1)
        mask = mask.repeat(distance.shape[0], 1, 1)
        distance = torch.mul(distance, mask)
        distance = torch.mean(distance)

        content_embeddings, _ = self.lstm_post(content_embeddings)

        #combining all stage sub-graphs’ representation  
        content_embeddings = self.pool_graph(content_embeddings).view(-1,
                                                                      (self.t_hidden_dim + self.st_hidden_dim) * 2)

        # mlp
        output = self.mlp(content_embeddings)
        return output, 1/distance, word_score, t_post_score, st_post_score
