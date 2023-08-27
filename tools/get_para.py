from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append("../")
import random
import argparse
from utils.util import create_samples, create_samples_mis
from config.config import Config
from train_MD import T_create_samples, to_cuda, get_model
import torch
from definition.MD_Dataset import MD_Dataset_para
from torch.utils.data import TensorDataset, DataLoader
import pickle

parser = argparse.ArgumentParser(description="Parameters of Misinformation Detection Experiment")
parser.add_argument("--train_rate", type=float, help="required", default=0.8)
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
parser.add_argument("--logfile", type=str, help="required", default="main_experiment.log") # 日志文件名
parser.add_argument("--server", type=str, help="required", default="local")  # 服务器位置（实验室/学校）
parser.add_argument("--rand", type=int, help="required", default=18)
parser.add_argument("--early", type=int, help="required", default=0)  # 默认不进行 及早检测 实验
parser.add_argument("--interval", type=int, help="required", default=0)  # 默认参数从 cnf.ini 中读取
parser.add_argument("--interval_num", type=int, help="required", default=0)  # 默认参数从 cnf.ini 中读取
parser.add_argument("--version", type=str, help="required", default="v0_linear_loss_para")
parser.add_argument("--gpu_num", type=str, help="required", default="2")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

cnf = Config(dataset=args.dataset, version=args.version, logfile=args.logfile, server=args.server,
             early=args.early, interval=args.interval, interval_num=args.interval_num)

def to_dataloader_mis(datafolder, origin_ids):
    dataset = MD_Dataset_para(datafolder, origin_ids)
    dataloader = DataLoader(dataset, batch_size=cnf.batch_size)
    return dataloader,dataset.order2id

def to_dataloader(data):
    id2order = {str(sample.origin_id): i for i, sample in enumerate(data)}
    order2id = {i: str(sample.origin_id) for i, sample in enumerate(data)}
    ids = torch.tensor([id2order[sample.origin_id] for sample in data], dtype=torch.long)
    content_embeddings = torch.stack([sample.content_embeddings for sample in data])
    user_embeddings = torch.stack([sample.user_embeddings for sample in data])
    adj = torch.stack([sample.adj for sample in data])
    graph_per_interval = torch.stack([sample.graph_per_interval for sample in data])
    content_mask = torch.stack([sample.content_mask for sample in data])
    post_mask_per_interval = torch.stack([sample.post_mask_per_interval for sample in data])
    label = torch.tensor([sample.label for sample in data])
    dataset = TensorDataset(content_embeddings, user_embeddings, adj, graph_per_interval,
                            content_mask, post_mask_per_interval, label, ids)
    dataloader = DataLoader(dataset, batch_size=cnf.batch_size)

    return dataloader, order2id

def get_samples_para():
    """
    获取处理后的样本数据，划分为 训练集和测试集
    :return:
    """
    if args.dataset in ["twitter15", "twitter16"]:
        samples = T_create_samples(cnf)
    elif args.dataset == "misinfdect":
        data = "data_" + str(cnf.max_seq_len) + "_" + str(cnf.max_post_num) \
               + "_" + str(cnf.interval) + "_" + str(cnf.interval_num)
        processed_datafolder = os.path.join(cnf.processed_data, data)
        if os.path.exists(processed_datafolder):
            files = os.listdir(processed_datafolder)
            origin_ids = [file.strip().split(".")[0] for file in files]
            dataloader, order2id = to_dataloader_mis(processed_datafolder, origin_ids)

            return dataloader, order2id
        else:
            create_samples_mis(cnf)
            sys.exit()
    else:
        samples = create_samples(cnf)

    dataloader, order2id = to_dataloader(samples)

    return dataloader, order2id

def load_best_model(model, cnf):
    model_name = "MD_t(0.8)_e(0)_para(3600_32)_v0_linear_loss_para_0.8802.pt"
    model_file = os.path.join(cnf.modelfolder, model_name)
    model.load_state_dict(torch.load(model_file))
    return model

def run():
    dataloader, order2id = get_samples_para()
    model = get_model()
    print(model)
    model = load_best_model(model, cnf)
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    model.eval()


    tqdm_data = tqdm(dataloader)
    Word_score = {}
    T_post_score = {}
    St_post_score = {}
    Graph = {}
    for batch_data in tqdm_data:
        order = batch_data[-1]
        content_embeddings, user_embeddings, adj, graph_per_interval, \
        content_mask, post_mask_per_interval, label = to_cuda(batch_data, cuda)
        out, dis, word_score, t_post_score, st_post_score = model(content_embeddings, user_embeddings, adj,
                                  graph_per_interval, content_mask, post_mask_per_interval)
        word_score = word_score.view(-1, cnf.max_post_num, cnf.max_seq_len).cpu().detach().numpy()
        t_post_score = t_post_score.view(-1, cnf.interval_num, cnf.max_post_num).cpu().detach().numpy()
        st_post_score = st_post_score.view(-1, cnf.interval_num, cnf.max_post_num).cpu().detach().numpy()
        graph_per_interval = graph_per_interval.cpu().numpy()
        order = order.numpy()

        for order_item, word_score_item in zip(order, word_score):
            mid = order2id[order_item]
            Word_score[mid] = list(word_score_item)

        for order_item, t_post_score_item in zip(order, t_post_score):
            mid = order2id[order_item]
            T_post_score[mid] = list(t_post_score_item)

        for order_item, st_post_score_item in zip(order, st_post_score):
            mid = order2id[order_item]
            St_post_score[mid] = list(st_post_score_item)

        for order_item, graph in zip(order, graph_per_interval):
            mid = order2id[order_item]
            Graph[mid] = list(graph)

    word_score_path = os.path.join(cnf.reprfolder, "word_score.pkl")
    with open(word_score_path,'wb') as f:
        pickle.dump(Word_score,f)

    t_post_score_path = os.path.join(cnf.reprfolder, "t_post_score.pkl")
    with open(t_post_score_path, 'wb') as f:
        pickle.dump(T_post_score, f)

    st_post_score_path = os.path.join(cnf.reprfolder, "st_post_score.pkl")
    with open(st_post_score_path, 'wb') as f:
        pickle.dump(St_post_score, f)

    graph_path = os.path.join(cnf.reprfolder, "graph.pkl")
    with open(graph_path, 'wb') as f:
        pickle.dump(Graph, f)

    print("ok")

run()
