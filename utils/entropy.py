# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 15:08
# @Author  : Naynix
# @File    : entropy.py
from utils.util import read_origin,read_retweets, time_interval
import argparse
from config.config import Config
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import os
import psutil
import threading
import time
parser = argparse.ArgumentParser(description="Parameters of Propagation Predicted Experiment")
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
parser.add_argument("--method", type=str, help="required", default="time")
parser.add_argument("--unit", type=int, help="required", default=900)
args = parser.parse_args()
cnf = Config(dataset=args.dataset, version="")



def paint(f_graph_entropy, nf_graph_entropy,color):
    for mid, ens in f_graph_entropy.items():
        X = [i for i in range(len(ens))]
        Y = ens
        plt.plot(X,Y, c=color[0])
    for mid, ens in nf_graph_entropy.items():
        X = [i for i in range(len(ens))]
        Y = ens
        plt.plot(X,Y, c=color[1])
    plt.show()
    print("ok")


def get_entropy_num(mids, rel):
    id_map = {id:i for i, id in enumerate(mids)}

    graph_entropy = [0]
    G = nx.Graph()
    G.add_node(0)

    orders = [i for i in range(1, len(id_map))]
    for order in orders:
        r_id = mids[order]
        p_id, _ = rel[r_id]
        p_num = id_map[p_id]
        G.add_edge(order, p_num)
        G.add_edge(p_num, order)
        degrees = np.array(G.degree())[:,1]
        degrees = degrees/G.number_of_edges()
        ens = -np.log(degrees)*degrees

        graph_entropy.append(np.sum(ens))

    return graph_entropy


def get_entropy_time(mids, rel, total_t):
    id_map = {id: i for i, id in enumerate(mids)}
    nunit = int(np.ceil(total_t/args.unit))
    # if hours > 24:
    #     hours = 24
    graph_entropy = []
    G = nx.Graph()
    G.add_node(0)

    n_units = [i for i in range(1, nunit+1)]
    order = 1
    i = 1
    for n_unit in n_units:
        while order < len(mids):
            r_id = mids[order]
            p_id, t = rel[r_id]
            if t/args.unit > n_unit:
                break
            p_num = id_map[p_id]
            G.add_edge(order, p_num)
            G.add_edge(p_num, order)
            order += 1

        if G.number_of_nodes() == 1:
            graph_entropy.append(0)
            continue
        degrees = np.array(G.degree())[:, 1]
        degrees = degrees / G.number_of_edges()
        ens = -np.log(degrees) * degrees

        graph_entropy.append(np.sum(ens))
    # if mids[0] == "580329886532968448":
    #     print(mids)
    return graph_entropy


def process_samples(Retweet, origin):
    graph_entropy = {}
    for mid, retweets in Retweet:
        if len(retweets) == 0:
            continue

        if mid not in origin:
            continue

        start = origin[mid]["date"]
        mids = [mid]
        rel = {}
        total_t = time_interval(start, retweets[-1]["date"])
        # if(total_t<):
        #     continue
        # if(total_t>10800):
        #     break
        for retweet in retweets:
            r_id = retweet["mid"]
            mids.append(r_id)
            p_id = retweet["parent"]
            cur_t = retweet["date"]
            seconds = time_interval(start, cur_t)
            rel[r_id] = (p_id, seconds)
        # ens = get_entropy_num(mids, rel)
        ens = get_entropy_time(mids, rel, total_t)
        graph_entropy[mid] = ens
        # if len(graph_entropy) == 50:
        #     break

    return graph_entropy

def save_pkl(data, file):
    with open(file, "wb") as f:
        pkl.dump(data, f)

def read_pkl(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data

def get_samples(cnf):
    repr_folder = cnf.reprfolder
    fake_file = os.path.join(repr_folder, f"f_degree_entropy_{args.method}.pkl")
    nonfake_file = os.path.join(repr_folder, f"nf_degree_entropy_{args.method}.pkl")

    # if os.path.exists(fake_file) and os.path.exists(nonfake_file):
    #     f_graph_entropy = read_pkl(fake_file)
    #     nf_graph_entropy = read_pkl(nonfake_file)
    #
    #     paint(f_graph_entropy, nf_graph_entropy, ["red", "blue"])

    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)
    f_graph_entropy = process_samples(fake, origin)
    nf_graph_entropy = process_samples(nonfake, origin)

    # save_pkl(f_graph_entropy, fake_file)
    # save_pkl(nf_graph_entropy, nonfake_file)
    # paint(f_graph_entropy, nf_graph_entropy, ["red","blue"])



def process_entropy(cnf):
    repr_folder = cnf.reprfolder
    fake_file = os.path.join(repr_folder, f"f_degree_entropy_{args.method}.pkl")
    nonfake_file = os.path.join(repr_folder, f"nf_degree_entropy_{args.method}.pkl")
    f_entropy = read_pkl(fake_file)
    nf_entropy = read_pkl(nonfake_file)
    # Y = f_entropy["A0AeVDf9l"]
    # X = [i for i in range(0, len(Y))]
    # plt.scatter(X, Y)
    # plt.show()
    print("ok")
def monitor_memory():
    while True:
        memory_info = psutil.virtual_memory()
        print(f"Used memory: {memory_info.used / (1024 ** 3):.2f} GB")
        # Add a suitable time delay between memory checks
        time.sleep(60)
def analyze_tags(cnf):
    file = os.path.join(cnf.datafolder, "tags.pkl")
    tags = read_pkl(file)

    print("ok")

memory_thread = threading.Thread(target=monitor_memory)
memory_thread.start()
start = time.time()
get_samples(cnf)
# process_entropy(cnf)
print(time.time()-start)
memory_thread.join()
# analyze_tags(cnf)
