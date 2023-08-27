# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 10:05
# @Author  : Naynix
# @File    : train_MD.py
from utils.util import  metric, save_result, save_model
from utils.util_v2 import create_samples_mis, create_samples
from utils.twitter15_16_util import T_create_samples
from config.config import Config
from definition.MD_Dynamic_net import MD_Dynamic_net
from definition.MD_Dataset import MD_Dataset
import argparse
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import psutil
import threading
import time
import os
import sys
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description="Parameters of Misinformation Detection Experiment")
parser.add_argument("--train_rate", type=float, help="required", default=0.8)
parser.add_argument("--dataset", type=str, help="required", default="new_pheme")
parser.add_argument("--logfile", type=str, help="required", default="main_experiment.log") # 日志文件名
parser.add_argument("--server", type=str, help="required", default="local")  # 服务器位置（实验室/学校）
parser.add_argument("--rand", type=int, help="required", default=18)
parser.add_argument("--early", type=int, help="required", default=3)  # 默认不进行 及早检测 实验
parser.add_argument("--interval", type=int, help="required", default=0)  # 默认参数从 cnf.ini 中读取
parser.add_argument("--interval_num", type=int, help="required", default=0)  # 默认参数从 cnf.ini 中读取
parser.add_argument("--version", type=str, help="required", default="entrophy")
parser.add_argument("--gpu_num", type=str, help="required", default="0")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

cnf = Config(dataset=args.dataset, version=args.version, logfile=args.logfile, server=args.server,
             early=args.early, interval=args.interval, interval_num=args.interval_num)


# def to_dataloader(data):
#     content_embeddings = torch.stack([sample.content_embeddings for sample in data])
#     user_embeddings = torch.stack([sample.user_embeddings for sample in data])
#     adj = torch.stack([sample.adj for sample in data])
#     graph_per_interval = torch.stack([sample.graph_per_interval for sample in data])
#     content_mask = torch.stack([sample.content_mask for sample in data])
#     post_mask_per_interval = torch.stack([sample.post_mask_per_interval for sample in data])
#     label = torch.tensor([sample.label for sample in data])
#     dataset = TensorDataset(content_embeddings, user_embeddings, adj, graph_per_interval,
#                                   content_mask, post_mask_per_interval, label)
#     dataloader = DataLoader(dataset, batch_size=cnf.batch_size)
#     return dataloader

def to_dataloader(data):
    dataset = [(sample.content_embeddings, sample.user_embeddings, sample.adj, sample.graph_per_interval,
                sample.content_mask, sample.post_mask_per_interval, sample.label) for sample in data]

    dataloader = DataLoader(dataset, batch_size=cnf.batch_size)
    return dataloader


def to_dataloader_mis(datafolder, origin_ids):
    dataset = MD_Dataset(datafolder, origin_ids)
    dataloader = DataLoader(dataset, batch_size=cnf.batch_size)
    return dataloader


def get_samples():
    """
    获取处理后的样本数据，划分为 训练集和测试集
    :return:
    """
    if args.dataset in ["twitter15", "twitter16"]:
        samples = T_create_samples(cnf)
    elif args.dataset == "misinfdect":
        data = "data_" + str(cnf.max_seq_len) + "_" + str(cnf.max_post_num) \
               + "_" + "entrophy" + "_" + str(cnf.interval_num)
        if args.early != 0:
            data = data + "_" + str(args.early)
        processed_datafolder = os.path.join(cnf.processed_data, data)
        if os.path.exists(processed_datafolder):
            files = os.listdir(processed_datafolder)
            origin_ids = [file.strip().split(".")[0] for file in files]
            trainids, testids = train_test_split(origin_ids, test_size=1 - args.train_rate, random_state=args.rand)
            valids, testids = train_test_split(testids, test_size=0.5, random_state=args.rand)
            train_dataloader = to_dataloader_mis(processed_datafolder, trainids)
            val_dataloader = to_dataloader_mis(processed_datafolder, valids)
            test_dataloader = to_dataloader_mis(processed_datafolder, testids)
            return train_dataloader, val_dataloader, test_dataloader
        else:
            create_samples_mis(cnf, args.early)
            sys.exit()
    else:
        samples = create_samples(cnf)

    traindata, testdata = train_test_split(samples, test_size=1 - args.train_rate, random_state=args.rand)
    valdata, testdata = train_test_split(testdata, test_size=0.5, random_state=args.rand)

    train_dataloader = to_dataloader(traindata)
    val_dataloader = to_dataloader(valdata)
    test_dataloader = to_dataloader(testdata)

    return train_dataloader, val_dataloader, test_dataloader

def to_cuda(batchdata, cuda):
    """
    数据转移至gpu
    :param batchdata:
    :param cuda:
    :return:
    """
    if cuda:
        content_embeddings = batchdata[0].cuda()
        user_embeddings = batchdata[1].cuda()
        adj = batchdata[2].cuda()
        graph_per_interval = batchdata[3].cuda()
        content_mask = batchdata[4].cuda()
        post_mask_per_interval = batchdata[5].cuda()
        labels = batchdata[6].cuda()
    else:
        content_embeddings = batchdata[0]
        user_embeddings = batchdata[1]
        adj = batchdata[2]
        graph_per_interval = batchdata[3]
        content_mask = batchdata[4]
        post_mask_per_interval = batchdata[5]
        labels = batchdata[6]
    return content_embeddings, user_embeddings, adj, graph_per_interval, content_mask, post_mask_per_interval, labels


def get_model():
    model = MD_Dynamic_net(input_dim=cnf.input_dim, user_dim=cnf.user_dim, hidden_dim=cnf.hidden_dim,
                               output_dim=cnf.output_dim, seq_len=cnf.max_seq_len, post_num=cnf.max_post_num,
                               interval_num=cnf.interval_num, lstm_layer_num=cnf.lstm_layer_num,
                               gat_layer_num=cnf.gat_layer_num, nheads=cnf.nheads, dropout=cnf.dropout,
                               alpha=cnf.alpha, pool=cnf.pool)
    return model


def get_res(batch_data, model, cuda):
    content_embeddings, user_embeddings, adj, graph_per_interval, \
    content_mask, post_mask_per_interval, label = to_cuda(batch_data, cuda)
    out, dis, _, _, _= model(content_embeddings, user_embeddings, adj,
                   graph_per_interval, content_mask, post_mask_per_interval)
    return out, label, dis


def test(model, dataloader, cuda, state):
    """
    模型测试
    :param model:
    :param dataloader:
    :param cuda:
    :return:
    """
    losses = []
    outs = []
    labels = []
    tq_test = tqdm(dataloader, desc="Iteration")
    with torch.no_grad():
        model.eval()
        train_reprs = []
        for batch_data in tq_test:
            out, label, _ = get_res(batch_data, model, cuda)
            loss = F.cross_entropy(out, label)
            tq_test.set_description("loss: " + str(loss.item()))
            losses.append(loss.item())
            outs.append(out)
            labels.append(label)

        test_loss = np.mean(losses)
        outs = torch.cat(outs, dim=0)
        labels = torch.cat(labels,dim=0)
        result = metric(outs, labels)

        cnf.logger.info("-" * 40 + state + ":" + "-" * 40)
        cnf.logger.info(state + " loss: " + str(test_loss))
        cnf.logger.info(str(result))

        print(state + ": loss = ", test_loss, "result: ", result)

    return result


def compare(res, best_res):
    if res["f1_1"] > best_res["f1_1"]:
        return 1, "f1_1"
    elif res["acc"] > best_res["acc"]:
        return 1, "acc"
    elif res["auc"] > best_res["auc"]:
        return 1, "auc"
    else:
        return 0, ""


def train():
    """
    模型训练
    :return:
    """

    train_dataloader, val_dataloader, test_dataloader = get_samples()
    model = get_model()
    print(model)
    group_para = [{"params": [para for name, para in model.named_parameters() if "gat" in name],
                   "lr":5e-4},
                  {"params": [para for name, para in model.named_parameters() if "mlp" in name],
                   "lr": 5e-3},
                  {"params": [para for name, para in model.named_parameters() if "gat" not in name and "mlp" not in name]}
                 ]

    # optimizer = Adam(group_para, lr=cnf.lr)
    optimizer = Adam(model.parameters(), lr=cnf.lr)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5, last_epoch=-1)
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    best_result = {
        "acc": 0.0,
        "prec_1": 0.0,
        "recall_1": 0.0,
        "f1_1": 0.0,
        "prec_0": 0.0,
        "recall_0": 0.0,
        "f1_0": 0.0,
        "auc": 0.0,
        "ap": 0.0
    }
    early_stop = 25  #实验早停设置
    cnt = 0
    print("*" * 40 + " start of train " + "*" * 40)
    cnf.logger.info("*" * 40 + " start of train " + "*" * 40)
    for epoch in range(cnf.epoch):
        cnf.logger.info("-" * 40 + "epoch: " + str(epoch) + "-" * 40)
        print("-" * 40 + "epoch: " + str(epoch) + "-" * 40)
        model.train()
        losses = []
        outs = []
        labels = []

        tq_train = tqdm(train_dataloader, desc="Iteration")
        for batch_data in tq_train:
            optimizer.zero_grad()
            out, label, dis = get_res(batch_data, model, cuda)
            loss = F.cross_entropy(out, label)
            # loss = torch.add(0.9 * loss, 0.1 * dis)
            loss.backward()
            optimizer.step()
            tq_train.set_description("loss: " + str(loss.item()))
            losses.append(loss.item())
            outs.append(out)
            labels.append(label)
        # scheduler.step()
        train_loss = np.mean(losses)
        outs = torch.cat(outs, dim=0)
        labels = torch.cat(labels, dim=0)
        result = metric(outs, labels)

        cnf.logger.info("-" * 40 + "train:" + "-" * 40)
        cnf.logger.info("train loss: " + str(train_loss))
        cnf.logger.info(str(result))
        print("train: loss = ", train_loss, "result: ", result)

        val_result = test(model, val_dataloader, cuda, "val")

        status, index = compare(val_result, best_result)
        if status:
            best_result[index] = val_result[index]
            cnt = 0
            save_model(model, cnf, args.early, args.train_rate, args.version, best_result)
            test_result = test(model, test_dataloader, cuda, "test")
            save_result(test_result, cnf, args.train_rate, args.early, args.version)
        else:
            cnt += 1
        if cnt >= early_stop:
            break

    print("*" * 40 + " end of train " + "*" * 40)
    cnf.logger.info("*" * 40 + "end of train" + "*" * 40)
    cnf.logger.info(str(best_result))

    del train_dataloader, test_dataloader
    del model, optimizer
    del best_result

def print_cnf():
    """
    日志打印模型参数
    :return:
    """
    cnf.logger.info(
        'train_rate: %f, input_dim: %d, hidden_dim: %d, interval: %d, interval_num: %d \n'
        'max_seq_len,:%d, max_post_num: %d,batch_size:%d, nheads: %d\n'
        'dropout:%f, pool:%s, alpha: %f, lr:%f'
        % (args.train_rate, cnf.input_dim, cnf.hidden_dim, cnf.interval, cnf.interval_num,
           cnf.max_seq_len, cnf.max_post_num, cnf.batch_size, cnf.nheads,
           cnf.dropout, cnf.pool, cnf.alpha, cnf.lr))
    print(
        'train_rate: %f, input_dim: %d, hidden_dim: %d, interval: %d, interval_num: %d \n'
        'max_seq_len,:%d, max_post_num: %d,batch_size:%d, nheads: %d\n'
        'dropout:%f, pool:%s, alpha: %f, lr:%f'
        % (args.train_rate, cnf.input_dim, cnf.hidden_dim, cnf.interval, cnf.interval_num,
           cnf.max_seq_len, cnf.max_post_num, cnf.batch_size, cnf.nheads,
           cnf.dropout, cnf.pool, cnf.alpha, cnf.lr))
def monitor_memory():
    while True:
        memory_info = psutil.virtual_memory()
        print(f"Used memory: {memory_info.used / (1024 ** 3):.2f} GB")
        # Add a suitable time delay between memory checks
        time.sleep(180)
if __name__ == "__main__":
    print_cnf()
    memory_thread = threading.Thread(target=monitor_memory)
    memory_thread.start()
    train()
    # Don't forget to join the thread at the end
    memory_thread.join()