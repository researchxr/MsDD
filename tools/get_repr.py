# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 14:54
# @Author  : Naynix
# @File    : get_repr.py
import argparse
import sys
sys.path.append('/mntc/yxy/MDPP')
from config.config import Config
from utils.util import create_samples, load_model, to_cuda
from definition.MD_Dynamic_net import MD_Dynamic_net
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

"""
加载已保存的MD模型，获取样本数据中传播树节点的表示向量
"""

parser = argparse.ArgumentParser(description="default")
parser.add_argument("--train_rate", type=float, help="required", default=0.8)
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
parser.add_argument("--server", type=str, help="required", default="local") # 服务器位置（实验室/学校）
parser.add_argument("--early", type=int, help="required", default=3) # 默认不进行 及早检测 实验
parser.add_argument("--interval", type=int, help="required", default=0) # 默认参数从 cnf.ini 中读取
parser.add_argument("--interval_num", type=int, help="required", default=0) # 默认参数从 cnf.ini 中读取
args = parser.parse_args()

cnf = Config(dataset=args.dataset, server=args.server, early=args.early,
             interval=args.interval, interval_num=args.interval_num)

def get_samples():
    """
    获取所有样本数据
    :return:
    """
    samples = create_samples(cnf)
    mid2num = {str(sample.origin_id): i for i, sample in enumerate(samples)}
    num2mid = {i: str(sample.origin_id) for i, sample in enumerate(samples)}
    ids = torch.tensor([mid2num[sample.origin_id] for sample in samples], dtype=torch.long)
    content_embeddings = torch.stack([sample.content_embeddings for sample in samples])
    user_embeddings = torch.stack([sample.user_embeddings for sample in samples])
    adj = torch.stack([sample.adj for sample in samples])
    graph_per_interval = torch.stack([sample.graph_per_interval for sample in samples])
    content_mask = torch.stack([sample.content_mask for sample in samples])
    post_mask_per_interval = torch.stack([sample.post_mask_per_interval for sample in samples])
    label = torch.tensor([sample.label for sample in samples])
    dataset = TensorDataset(content_embeddings, user_embeddings, adj, graph_per_interval,
                                  content_mask, post_mask_per_interval, label, ids)
    dataloader = DataLoader(dataset, batch_size=cnf.batch_size)

    return dataloader, num2mid


def get_embeddings():
    model = MD_Dynamic_net(input_dim=cnf.input_dim, user_dim=cnf.user_dim, hidden_dim=cnf.hidden_dim,
                           output_dim=cnf.output_dim, seq_len=cnf.max_seq_len, post_num=cnf.max_post_num,
                           interval_num=cnf.interval_num, lstm_layer_num=cnf.lstm_layer_num,
                           gat_layer_num=cnf.gat_layer_num, nheads=cnf.nheads, dropout=cnf.dropout,
                           alpha=cnf.alpha, pool=cnf.pool)
    model = load_model(model, cnf, args.early, args.train_rate)
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    model.eval()
    data_loader, num2mid = get_samples()
    Reprs = {}
    tqdm_data = tqdm(data_loader)
    for batch_data in tqdm_data:
        nums = batch_data[-1]
        content_embeddings, user_embeddings, adj, graph_per_interval, \
        content_mask, post_mask_per_interval, label = to_cuda(batch_data, cuda)
        _, reprs = model(content_embeddings, user_embeddings, adj,
                        graph_per_interval, content_mask, post_mask_per_interval)

        nums = nums.numpy()
        reprs = reprs.detach().cpu().numpy()

        for i, repr in zip(nums, reprs):
            Reprs[num2mid[i]] = repr

    file_name = "MD_repr(" + str(args.train_rate) + ") e(" + str(args.early) + ") para(" + \
                 str(cnf.interval) + " " + str(cnf.interval_num) + ").pkl"
    repr_file = os.path.join(cnf.reprfolder, file_name)
    with open(repr_file, 'wb') as f:
        pickle.dump(Reprs, f)


get_embeddings()