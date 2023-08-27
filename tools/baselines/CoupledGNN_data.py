# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 16:41
# @Author  : Naynix
# @File    : CoupledGNN_data.py
import argparse
from config.config import Config
from utils.PP_util import read_origin, read_retweets, read_user_embeddings
from utils.util import time_interval
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="Parameters of Misinformation Detection Experiment")
parser.add_argument("--dataset", type=str, help="required", default="pheme")
parser.add_argument("--train_rate", type=float, help="required", default=0.8)
parser.add_argument("--early", type=int, help="required", default=3)  # 默认不进行 及早检测 实验
args = parser.parse_args()

cnf = Config(dataset=args.dataset)


def construct_graph(M_usermap, R_follow, R_retweet, early):
    graph = {}
    for item in R_follow:
        edge_from = M_usermap[item[0]] - 1
        edge_to = M_usermap[item[1]] - 1
        if edge_from not in graph:
            graph[edge_from] = set()
            graph[edge_from].add(edge_to)
        else:
            graph[edge_from].add(edge_to)

    for item in R_retweet:
        if item[2] <= early * 3600:
            edge_from = M_usermap[item[0]] - 1
            edge_to = M_usermap[item[1]] - 1
            if edge_from not in graph:
                graph[edge_from] = set()
                graph[edge_from].add(edge_to)
            else:
                graph[edge_from].add(edge_to)

    for edge_from, edge_to in graph.items():
        graph[edge_from] = list(edge_to)

    return graph


def construct_samples(origin, fake, M_usermap, early):

    Xs = []
    ys = []
    for mid, retweets in fake:
        source = origin[mid]
        start = source["date"]
        source_uid = M_usermap[source["user"]["_id"]] - 1
        sample_x = list()
        uid_x = set()
        sample_x.append((0.0, source_uid))
        uid_x.add(source_uid)

        sample_y = set()
        sample_y.add(source_uid)

        for retweet in retweets:
            end = retweet["date"]
            uid = M_usermap[retweet["user"]["_id"]] - 1
            seconds = time_interval(start, end)
            hours = float(np.floor(seconds/3600))
            if hours < early and uid not in uid_x:
                sample_x.append((hours, uid))
                uid_x.add(uid)
            sample_y.add(uid)

        Xs.append(sample_x)
        ys.append(sample_y)
    num = 0
    for sample_x, sample_y in zip(Xs, ys):
        if len(sample_y) <= len(sample_x):
            num += 1
    return Xs, ys


def save_data():

    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)

    with open(cnf.usermapfile, "rb") as f:
        M_usermap = pickle.load(f)
    with open(cnf.followfile, "rb") as f:
        R_follow = pickle.load(f)
    with open(cnf.retweetfile, "rb") as f:
        R_retweet = pickle.load(f)

    user_embeddings = read_user_embeddings(cnf.user_embedding_file, M_usermap)
    user_embeddings = user_embeddings.cpu().numpy()
    user_embeddings = list(map(list, user_embeddings))
    graph = construct_graph(M_usermap, R_follow, R_retweet, args.early)
    Xs, ys = construct_samples(origin, fake, M_usermap, args.early)
    train_x, test_x, train_y, test_y = train_test_split(Xs, ys, train_size=args.train_rate, random_state=0)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, train_size=0.5, random_state=0)

    folder = "./CoupledGNN"
    dataset_str = "pheme"
    if not os.path.exists(folder):
        os.makedirs(folder)

    names = ['train.x', 'train.y', 'val.x', 'val.y', 'test.x', 'test.y', 'graph', 'features']
    objects = [train_x, train_y, val_x, val_y, test_x, test_y, graph, user_embeddings]
    for i in range(len(names)):
        file_path = os.path.join(folder, "ind.{}.{}".format(dataset_str, names[i]))
        with open(file_path, 'wb') as f:
            pickle.dump(objects[i], f)

    print("ok")

save_data()