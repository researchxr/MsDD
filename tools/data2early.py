# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 19:32
# @Author  : Naynix
# @File    : data2early.py

"""
处理获得前 n 小时的推文传播数据，主要为虚假信息的早期检测任务以及虚假信息的传播趋势预测任务提供数据。
"""
import sys
import os
# 获取当前脚本所在的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加外部文件夹的路径到sys.path
external_folder_path = os.path.join(current_dir, '/nfs/users/lijinze/whd/MDPP_versions/entrophy')
sys.path.append(external_folder_path)

from config.config import Config
from utils.util import read_origin, read_retweets, time_interval
import argparse
import os
import json


parser = argparse.ArgumentParser(description="default")
parser.add_argument("--dataset", type=str, help="required", default="new_pheme")
parser.add_argument("--early", type=int, help="required", default=3)

args = parser.parse_args()

cnf = Config(dataset=args.dataset)


def get_early_Retweets(origin, Retweets, early):
    early_Retweets = {}
    for mid, retweets in Retweets:

        start_t = origin[mid]["date"]

        end = 0
        for retweet in retweets:
            t = retweet["date"]
            interval = time_interval(start_t, t) / 3600.0
            if interval > early:
                continue
            end = end + 1
        if end != 0:
            early_Retweets[mid] = retweets[0:end]

    return early_Retweets


def save_Retweets(folder, early_Retweets):

    for mid, retweets in early_Retweets.items():
        file = mid + '.json'
        filepath = os.path.join(folder, file)
        with open(filepath, "w") as f:
            json.dump(retweets, f, indent=4)

def data2early():
    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)

    fake_early_Retweets = get_early_Retweets(origin, fake, args.early)
    nonfake_early_Retweets = get_early_Retweets(origin, nonfake, args.early)

    fake_early_folder = os.path.join(cnf.datafolder, "early/fake_" + str(args.early))
    nonfake_early_folder = os.path.join(cnf.datafolder, "early/nonfake_" + str(args.early))
    if not os.path.exists(fake_early_folder):
        os.makedirs(fake_early_folder)
    if not os.path.exists(nonfake_early_folder):
        os.makedirs(nonfake_early_folder)
    save_Retweets(fake_early_folder, fake_early_Retweets)
    save_Retweets(nonfake_early_folder, nonfake_early_Retweets)

data2early()
