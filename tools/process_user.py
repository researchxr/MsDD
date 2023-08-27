# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 19:42
# @Author  : Naynix
# @File    : process_user

import pickle
import argparse
from utils.util import read_origin, read_retweets, time_interval
from utils.PP_util_1 import read_popularity, construct_ego_network
from config.config import Config
import os

parser = argparse.ArgumentParser(description="Parameters of Propagation Predicted Experiment")
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
parser.add_argument("--early", type=int, help="required", default=3)
args = parser.parse_args()
cnf = Config(dataset=args.dataset, early=args.early)

def get_R_retweet(fake, origin, early):
    R_retweet = []
    for mid, retweets in fake:
        start = origin[mid]["date"]
        for item in retweets:
            parent = item["parent"]
            user = item["user"]["_id"]
            end = item["date"]
            seconds = time_interval(start, end)
            R_retweet.append((user, parent, seconds))


    return R_retweet

def process_user():
    orgin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    if os.path.exists(cnf.retweetfile):
        with open(cnf.retweetfile, "rb") as f:
            R_retweet = pickle.load(f)
    else:
        R_retweet = get_R_retweet(fake, orgin, args.early)

    R = {}
    for item in R_retweet:
        edge_from = item[0]
        edge_to = item[1]
        if edge_from not in R:
            R[edge_from] = set()
            R[edge_from].add(edge_to)
        else:
            R[edge_from].add(edge_to)

    with open(cnf.retweetfile, "wb") as f:
        pickle.dump(R_retweet, f)
    pop_final, pop_user_early, pop_user_final = read_popularity(cnf.pop_file, cnf.pop_user_early_file,
                                                                cnf.pop_user_final_file)
    ego_networks = construct_ego_network(pop_user_early, R)
    print("ok")




process_user()