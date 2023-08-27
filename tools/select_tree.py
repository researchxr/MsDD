# -*- coding: utf-8 -*-
# @Time    : 2022/3/7 15:31
# @Author  : Naynix
# @File    : select_tree.py
"""
统计树的深度
"""
import json
import os
import pickle
import queue
import datetime
import argparse
from config.config import Config
parser = argparse.ArgumentParser(description="Parameters of Misinformation Detection Experiment")
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
parser.add_argument("--server", type=str, help="required", default="local")  # 服务器位置（实验室/学校）
parser.add_argument("--version", type=str, help="required", default="v0_linear_loss_para")
args = parser.parse_args()

cnf = Config(dataset=args.dataset, version=args.version)

def cal_time(start,end):
    start = datetime.datetime.strptime(start,'%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    h = (end-start).total_seconds()
    return h

def open_retweet(type):
    relation = {}
    times = {}
    if type =="fake":
        path = cnf.fakefolder
    else:
        path = cnf.nonfakefolder

    files = os.listdir(path)
    for file in files:
        origin_id = file.split('.')[0]
        filepath = os.path.join(path, file)

        with open(filepath, 'r') as f:
            infos = json.load(f)
            children = {}
            if (len(infos) == 0):
                break
            start = infos[0]["date"]
            end = infos[-1]["date"]
            h = cal_time(start,end) / 3600.0

            for info in infos:
                if info["parent"] == "":
                    p_id = origin_id
                else:
                    p_id = info["parent"]
                if p_id in children:
                    children[p_id].append(info["mid"])
                else:
                    children[p_id] = [info["mid"]]

        relation[origin_id] = children
        times[origin_id] = h
    return relation,times


def calculate(type):
    relations,times = open_retweet(type)
    tree_info = {}
    for key, relation in relations.items():

        q = queue.Queue()
        depth = 1
        post_number = 0
        node = (key, depth)
        q.put(node)

        while not q.empty():
            cur = q.get()
            post_number = post_number + 1
            if cur[0] not in relation:
                continue
            for child in relation[cur[0]]:
                node = (child, cur[1] + 1)
                if (cur[1] + 1 > depth):
                    depth = cur[1] + 1
                q.put(node)
        if depth != 1:
            tree_info[key] = (depth, post_number)

    return tree_info,times

fake_tree_info, times_fake = calculate('fake')
nonfake_tree_info, times_nonfake = calculate('nonfake')
selected = []
for key,info in fake_tree_info.items():
    if info[0]>=3 and info[0] <= 24 and info[1]>=16 and info[1]<=72 and times_fake[key] > 6:
        selected.append(key)
for key,info in nonfake_tree_info.items():
    if info[0]>=3 and info[0] <= 24 and info[1]>=16 and info[1]<=72 and times_nonfake[key] > 6:
        selected.append(key)

dest_file = os.path.join(cnf.reprfolder, "selected_tree.pkl")
with open(dest_file,'wb') as f:
    pickle.dump(selected,f)
print("ok")

