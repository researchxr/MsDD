# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 11:32
# @Author  : Naynix
# @File    : visualize.py
import argparse
import pickle
import os
from config.config import Config
from utils.util import read_origin
import numpy as np
import seaborn as sns
import json
from tqdm import tqdm
from itertools import chain

parser = argparse.ArgumentParser(description="Parameters of Misinformation Detection Experiment")
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
parser.add_argument("--server", type=str, help="required", default="local")  # 服务器位置（实验室/学校）
parser.add_argument("--version", type=str, help="required", default="v0_linear_loss_para")
parser.add_argument("--gpu_num", type=str, help="required", default="2")
parser.add_argument("--color", type=str, help="required", default="dodgerblue")
parser.add_argument("--word_color", type=str, help="required", default="magenta")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
cnf = Config(dataset=args.dataset, version=args.version)

def read_pkl(file):
    with open(file,"rb") as f:
        data = pickle.load(f)

    return data

def read_retweets(retweet_folder):
    retweet = {}
    files = os.listdir(retweet_folder)
    for file in files:
        mid = file.split('.')[0]
        file_path = os.path.join(retweet_folder, file)
        with open(file_path, 'r') as f:
            infos = json.load(f)
            contents = []
            for info in infos:
                user = {}
                user["_id"] = info["user"]["_id"]
                user["name"] = info["user"]["name"]
                user["verified"] = info["user"]["verified"]
                user["verified_reason"] = info["user"]["verified_reason"]
                user["verified_type"] = info["user"]["verified_type"]
                user["description"] = info["user"]["description"]
                user["gender"] = info["user"]["gender"]
                user["followers_count"] = info["user"]["followers_count"]
                user["followees_count"] = info["user"]["followees_count"]

                content = {}
                content["mid"] = info["mid"]
                content["text_seg"] = info["text_seg"]
                content["text"] = info["text"]
                content["date"] = info["date"]
                content["user"] = user
                if info["parent"] == "":
                    content["parent"] = mid
                else:
                    content["parent"] = info["parent"]
                contents.append(content)
            retweet[mid] = contents

    return retweet


def normalization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    data = (data-mu) / sigma
    _range = np.max(data) - np.min(data)

    return (data-np.min(data))/_range


def g2rel(graph):
    rel = {}
    ids = [i for i in range(len(graph))]

    for id in ids:
        graph[id][id] = 0

    for id, line in zip(ids, graph):
        nodes = [node for node, val in zip(ids, line) if val ==1]
        for node in nodes:
            if node not in rel:
                rel[node] = [id]
            else:
                rel[node].append(id)

    posts = set()
    for p_id, c_ids in rel.items():
        posts.add(p_id)
        for id in c_ids:
            posts.add(id)

    return rel, len(posts)

def find_children(p_id, rel, colors, text):

    try:
        if p_id not in rel:
           return {
               "name":p_id,
               "itemStyle":{
                   "color":colors[p_id],
                   "borderWidth":0
               },
               "value":p_id
           }
    except Exception as e:
        print(e)

    children = []
    for c_id in rel[p_id]:
        c_des = find_children(c_id, rel, colors, text)
        children.append(c_des)
    return {"name": p_id,
            "textStyle":{
                "color": colors[p_id]
            },
            "children": children,
            "itemStyle": {
                    "color": colors[p_id],
                    "borderWidth": 0
                        },
            "value": p_id }


def to_json(mid, rel, t_score, st_score, order, text):

    if st_score.shape[0] == 0:
        return

    max_post_score = np.max(st_score)
    min_post_score = np.min(st_score)
    _range = max_post_score - min_post_score
    colors = []
    for score in st_score:
        if _range != 0:
            p_attn = (score - min_post_score) / _range
        else:
            p_attn = 0.5
        color = '#%02X%02X%02X' % (255, int(200*(1-p_attn)), int(200*(1-p_attn)))
        colors.append(color)
    graph = find_children(0, rel, colors, text)

    output_folder = os.path.join(cnf.reprfolder, "figures", mid)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    json_file = os.path.join(output_folder, f"{mid}_{order}.json")
    with open(json_file, "w") as f:
        json.dump(graph, f, indent=4)

def paint_word(mid, word_score, post_score, text):
    sentence_num = len(text)

    post_score = post_score[:sentence_num]
    max_post_score = np.max(post_score)
    min_post_score = np.min(post_score)
    _range_post = max_post_score - min_post_score
    word_score = np.array(word_score)
    # word_score = normalization(word_score)
    # post_score = normalization(post_score)
    s_html = []
    for s_order in range(0, sentence_num):
        if _range_post != 0:
            p_attn = (post_score[s_order] - min_post_score) / _range_post
        else:
            p_attn = 0.5
        post_color = '#%02X%02X%02X' % (int(200 * (1 - p_attn)), 255, 255)
        if s_order == 0:  # 原微博
            w_html = [f'<span style="background-color:{post_color}" >origin:</span>']
        else:
            w_html = [f'<span style="background-color:{post_color}">retweet {s_order}:</span>']
        words = text[s_order].split(" ")
        if len(words) ==0:
            continue
        s_colors = word_score[s_order][:len(words)]
        max_word_score = np.max(s_colors)
        min_word_score = np.min(s_colors)
        _range = max_word_score - min_word_score

        for w_order in range(0, len(words)):
            # if w_order >= word_score.shape[1]:
            #     break
            if _range != 0:
                w_attn = (s_colors[w_order]-min_word_score)/_range
            else:
                w_attn = 0.5
            word_color = '#%02X%02X%02X' % (255, int(200*(1-w_attn)),255)
            html = f'<span style="background-color:{word_color}" >{words[w_order]}</span>'
            w_html.append(html)
        w_html = "&nbsp".join(w_html)

        w_html = f'<div style="margin:8px">{w_html}<br><div>'
        s_html.append(w_html)
    str = "".join(s_html)
    print(str)
    return s_html



def paint(mid, t_score, st_score, graphs, text):

    for order in range(0, cnf.interval_num):
        graph = graphs[order]
        rel, post_num = g2rel(graph)
        t_sc = t_score[order][:post_num]
        st_sc = st_score[order][:post_num]
        to_json(mid, rel, t_sc, st_sc, order, text)


def cal_std(selected_mids, fake, word_score):
    v_stds = []
    for mid, retweets in tqdm(fake.items()):
        if mid not in selected_mids:
            continue

        m_word_score = word_score[mid][0]
        v_std = np.sqrt(np.var(m_word_score))
        v_stds.append((mid, v_std))
    v_stds = sorted(v_stds, key=lambda x: x[1])
    for item in v_stds:
        print(item)

def generate_figure(fake, selected_mids, word_score, t_post_score, st_post_score, graph):
    for mid, retweets in tqdm(fake.items()):
        if mid not in selected_mids:
            continue
        text = [origins[mid]["text_seg"]]
        for retweet in retweets:
            text.append(retweet["text_seg"])
        # new_text = []
        # for item in text:
        #     words = item.split(" ")
        #     words = list(chain(
        #         *[words[i:i + 6] + ['\n'] if len(words[i:i + 6]) == 6 else words[i:i + 6]
        #           for i in range(0, len(words), 6)]))
        #     words = " ".join(words)
        #     new_text.append(words)

        m_graphs = graph[mid]
        # m_word_score = word_score[mid]
        m_t_post_score = t_post_score[mid]
        m_st_post_score = st_post_score[mid]
        paint(mid, m_t_post_score, m_st_post_score, m_graphs, text)
        # paint_word(mid, m_word_score, m_st_post_score[-1], text)

if __name__=="__main__":

    # figure_folder = os.path.join(cnf.reprfolder, "figures")
    # mids = os.listdir(figure_folder)
    # for mid in mids:
    #     print(mid)

    word_score_file = os.path.join(cnf.reprfolder, "word_score.pkl")
    t_post_score_file = os.path.join(cnf.reprfolder, "t_post_score.pkl")
    st_post_score_file = os.path.join(cnf.reprfolder, "st_post_score.pkl")
    graph_file = os.path.join(cnf.reprfolder, "graph.pkl")
    selected_file = os.path.join(cnf.reprfolder, "selected_tree.pkl")

    word_score = read_pkl(word_score_file)
    t_post_score = read_pkl(t_post_score_file)
    st_post_score= read_pkl(st_post_score_file)
    graph = read_pkl(graph_file)
    selected_mids = read_pkl(selected_file)

    origins = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)

    generate_figure(fake, selected_mids, word_score,
                    t_post_score, st_post_score, graph)

    # cal_std(selected_mids, fake, word_score)
    # mid = "z0LMt11LS"
    # text = [origins[mid]["text_seg"]]
    # retweets = fake[mid]
    # for retweet in retweets:
    #     text.append(retweet["text_seg"])
    # m_graphs = graph[mid]
    # m_word_score = word_score[mid]
    # m_t_post_score = t_post_score[mid]
    # m_st_post_score = st_post_score[mid]
    # # paint(mid, m_t_post_score, m_st_post_score, m_graphs, text)
    # paint_word(mid, m_word_score, m_st_post_score[-1], text)




    print("ok")



