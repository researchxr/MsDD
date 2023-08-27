# -*- coding: utf-8 -*-
# @Time    : 2021/11/19 10:10
# @Author  : Naynix
# @File    : WordEncoder.py
import torch
from torch import nn
import torch.nn.functional as F
import math


class AttAggregate(nn.Module):
    def __init__(self, input_dim, dropout):
        super(AttAggregate, self).__init__()
        self.input_dim = input_dim
        self.dropout =dropout
        self.att = nn.Linear(self.input_dim, 1)
        self.reset_para()

    def reset_para(self):
        gain = nn.init.calculate_gain('relu')
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)

    def forward(self, content_embeddings, content_mask):
        """

        :param content_embeddings: N * seq_len *input_dim
        :param content_mask: N*seq_len
        :return: N*input_dim
        """
        _, seq_len, _ = content_embeddings.shape
        score = self.att(content_embeddings.view(-1, seq_len, self.input_dim)).view(-1, seq_len)
        score = torch.mul(score, 1.0 / math.sqrt(float(self.input_dim)))

        content_mask = content_mask.masked_fill(content_mask == 0, -1e9).view(-1, seq_len)
        score = torch.add(score, content_mask)
        score = F.softmax(score, dim=-1).view(-1, seq_len, 1)
        content_embeddings = torch.bmm(content_embeddings.permute(0, 2, 1), score)

        content_embeddings = F.elu(content_embeddings)
        content_embeddings = F.dropout(content_embeddings, p=self.dropout)

        return content_embeddings, score


class WordEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layer_num, dropout):
        super(WordEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layer_num = lstm_layer_num
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim // 2,
                            num_layers=self.lstm_layer_num, batch_first=True, bidirectional=True)
        self.att_token = AttAggregate(input_dim=hidden_dim, dropout=dropout)

    def forward(self, content_embeddings, content_mask):
        """

        :param content_embeddings: N * seq_len *input_dim
        :param content_mask: N*seq_len
        :return: N*input_dim
        """
        content_embeddings, _ = self.lstm(content_embeddings)
        content_embeddings, word_score = self.att_token(content_embeddings, content_mask)
        return content_embeddings, word_score