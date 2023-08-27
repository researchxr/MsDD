# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 19:30
# @Author  : Naynix
# @File    : PP_popularity_true.py
from config.config import Config
from utils.util import read_origin, read_retweets, time_interval
import argparse
import os
import pickle


parser = argparse.ArgumentParser(description="default")
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
parser.add_argument("--early", type=int, help="required", default=3)

args = parser.parse_args()

cnf = Config(dataset=args.dataset)

def generate_popularity_true_file(early):
    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)

    pop_final = {}
    pop_user_final = {}
    pop_user_early = {}

    for mid, retweets in fake:
        num = len(retweets)
        pop_final[mid] = num + 1
        item_early = {}
        item_final = {}
        source = origin[mid]
        source_uid = source["user"]["_id"]
        start = source["date"]
        item_early[source_uid] = 1
        item_final[source_uid] = 1

        for retweet in retweets:
            uid = retweet["user"]["_id"]
            end = retweet["date"]
            hours = time_interval(start, end) / 3600.0
            if uid not in item_final:
                item_final[uid] = 1
            else:
                item_final[uid] += 1
            if hours <= early:
                if uid not in item_early:
                    item_early[uid] = 1
                else:
                    item_early[uid] += 1

        pop_user_final[mid] = item_final
        pop_user_early[mid] = item_early

    pop_final_file = os.path.join(cnf.truefolder, "pop_final.pkl")
    with open(pop_final_file, "wb") as f:
        pickle.dump(pop_final, f)

    pop_user_final_file = os.path.join(cnf.truefolder, "pop_user_final.pkl")
    with open(pop_user_final_file, "wb") as f:
        pickle.dump(pop_user_final, f)

    pop_user_early_file = os.path.join(cnf.truefolder, "pop_user_early_" + str(early) + ".pkl")
    with open(pop_user_early_file, "wb") as f:
        pickle.dump(pop_user_early, f)

generate_popularity_true_file(args.early)
