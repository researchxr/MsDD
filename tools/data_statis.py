# -*- coding: utf-8 -*-
# @Time    : 2022/2/15 10:37
# @Author  : Naynix
# @File    : data_statis.py

# from tools.data_analysis import CDF

from utils.util import read_origin, read_retweets, time_interval
from config.config import Config
import numpy as np
import argparse
import queue
import pandas as pd
import os
parser = argparse.ArgumentParser(description="default")
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
args = parser.parse_args()

cnf = Config(dataset=args.dataset, version="")


def tweet_num(Retweets):
    tweet_num = 0
    for mid, retweets in Retweets:
        tweet_num += len(retweets) + 1
    return tweet_num

def average_tweet_num(Retweets):
    tweet_num_list = []
    for mid, retweets in Retweets:
        tweet_num_list.append(len(retweets) + 1)
    return tweet_num_list, np.average(tweet_num_list)


def user_num(origin, Retweets):
    user = set()
    for mid, retweets in Retweets:
        user.add(origin[mid]["user"]["_id"])
        for retweet in retweets:
            user.add(retweet["user"]["_id"])
    return user


def average_time_span(origin, Retweets):

    times = []
    for mid, retweets in Retweets:
        if len(retweets) == 0:
            continue
        start = origin[mid]["date"]
        end = retweets[-1]["date"]
        t = time_interval(start, end)/3600.0
        times.append(t)
    return times, np.average(times)

def get_Relation(Retweets):
    Relation = {}
    leaf_num = []
    for mid, retweets in Retweets:
        children = {}
        if len(retweets) == 0:
            continue
        for retweet in retweets:
            if retweet["parent"] == "":
                parent_id = mid
            else:
                parent_id = retweet["parent"]
            if parent_id in children:
                children[parent_id].append(retweet["mid"])
            else:
                children[parent_id] = [retweet["mid"]]
        Relation[mid] = children
        leaf_num.append(len(retweets) + 1 - len(children))
    return Relation, leaf_num


def get_Tree_info(Retweets):
    Relation, leaf_num = get_Relation(Retweets)
    Tree_info = []
    for mid, relation in Relation.items():

        q = queue.Queue()
        depth = 1
        node = (mid, depth)
        q.put(node)

        while not q.empty():
            cur = q.get()
            if cur[0] not in relation:
                continue
            for child in relation[cur[0]]:
                node = (child, cur[1] + 1)
                if (cur[1] + 1 > depth):
                    depth = cur[1] + 1
                q.put(node)
        Tree_info.append(depth)
    # max_depth = np.max(Tree_info)
    average_depth = np.average(Tree_info)
    average_leaf_num = np.average(leaf_num)
    return Tree_info, leaf_num, average_depth, average_leaf_num



def BaseInfo(origin, fake, nonfake):
    info = {}
    f_samples_num = len(fake)
    nf_samples_num = len(nonfake)
    total_samples_num = f_samples_num + nf_samples_num

    f_tweet_num = tweet_num(fake)
    nf_tweet_num = tweet_num(nonfake)
    total_tweet_num = f_tweet_num + nf_tweet_num

    f_user_set = user_num(origin, fake)
    nf_user_set = user_num(origin, nonfake)
    user_set = f_user_set.union(nf_user_set)
    total_user_num = len(user_set)

    _, f_average_tweet_num = average_tweet_num(fake)
    _, nf_average_tweet_num = average_tweet_num(nonfake)
    total_average_tweet_num = (f_average_tweet_num + nf_average_tweet_num) / 2.0

    _, _, f_average_depth, f_average_leaf_num = get_Tree_info(fake)
    _, _, nf_average_depth, nf_average_leaf_num= get_Tree_info(nonfake)
    total_average_depth = (f_average_depth + nf_average_depth) / 2.0
    total_average_leaf_num = (f_average_leaf_num + nf_average_leaf_num) / 2.0

    _, f_average_time_span = average_time_span(origin, fake)
    _, nf_average_time_span = average_time_span(origin, nonfake)
    total_average_time_span = (f_average_time_span + nf_average_time_span) / 2.0

    info["dataset"] = args.dataset
    info["samples_num"] = total_samples_num
    info["f_samples_num"] = f_samples_num
    info["nf_samples_num"] = nf_samples_num
    info["tweet_num"] = total_tweet_num
    info["user_num"] = total_user_num
    info["average_tweet_num"] = total_average_tweet_num
    info["average_depth"] = total_average_depth
    info["average_leaf_num"] = total_average_leaf_num
    info["average_time_span"] = total_average_time_span
    print(info)

    df = pd.DataFrame(info, index=[0])
    info_file = "baseinfo.csv"
    if os.path.exists("baseinfo.csv"):
        df.to_csv(info_file, mode='a', header=False, index=False)
    else:
        df.to_csv(info_file, mode='a', header=True, index=False)


if __name__ == "__main__":
    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)
    BaseInfo(origin, fake, nonfake)

    print("ok")