# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 9:39
# @Author  : Naynix
# @File    : util_v2.py
import os
import pickle as pkl
from definition.Vocabulary import Vocabulary
from definition.InputSample import MDInputSample as InputSample
from utils.util import read_origin, read_retweets, pad_sentence, pad_post
from utils.util import token2embed, user2embed, construct_relation, time_interval
import torch
from torch import nn as nn
from tqdm import tqdm
import numpy as np

def read_pkl(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data

def analyze_tags(cnf):
    file = cnf.tagsfile
    tags = read_pkl(file)
    new_tags = {}
    for k,v in tags.items():
        new_tags[k] = [item[0] for item in v]
    return new_tags


def to_graphs(adj, dates, post_mask, interval_num, tags, tunit, max_post_num):
    """
    根据给定的时间间隔,得到不同时间阶段的关系的表示
    :param adj: 原微博和各转发微博的关系表示矩阵
    :param dates: 原微博和各转发微博的发布时间
    :param post_mask: 节点掩码
    :param interval: 时间间隔长度
    :param interval_num: 时间间隔个数
    :param max_post_num: 最大节点数目
    :return:
    """
    t_start = dates[0]
    time_intervals = []
    for date in dates:
        time_intervals.append(time_interval(t_start,date))
    time_intervals = np.array(time_intervals)
    graph_per_interval = []
    post_mask_per_interval = []
    adj = adj.transpose(1, 0)

    if len(tags) >= interval_num:
        tags = tags[:interval_num-1]

    for i in tags:
        mask = torch.tensor(time_intervals < tunit * (i+1), dtype=torch.float32)
        if mask.shape[0] < max_post_num:
            mask = torch.cat([mask, torch.zeros(max_post_num-mask.shape[0],dtype=torch.float32)], dim=0)
        else:
            mask = mask[:max_post_num]
        post_mask_per_interval.append(mask)
        mask = mask.repeat(max_post_num).view(max_post_num, max_post_num)
        graph = torch.mul(adj, mask).transpose(1, 0)
        graph_per_interval.append(graph)
    adj = adj.transpose(1, 0)
    graph_per_interval.append(adj)
    post_mask_per_interval.append(post_mask)


    while len(graph_per_interval) < interval_num:
        graph_per_interval.append(adj)
        post_mask_per_interval.append(post_mask)

    if len(graph_per_interval) > interval_num or len(post_mask_per_interval)>interval_num:
        print("len:", len(graph_per_interval), len(post_mask_per_interval))

    return torch.stack(graph_per_interval), torch.stack(post_mask_per_interval)


def construct_samples(retweets, **kwargs):
    """
    处理样本数据并封装
    :param retweets: 各个原微博的转发微博
    :param kwargs:
    :return:
    """
    origin = kwargs["origin"]
    para = kwargs["para"]
    vocab = kwargs["vocab"]
    Tags = kwargs["Tags"]
    label = kwargs["label"]

    max_seq_len = para["max_seq_len"]
    max_post_num = para["max_post_num"]
    interval = para["interval"]
    interval_num = para["interval_num"]
    input_dim = para["input_dim"]
    tunit = para["tunit"]

    pretrained_embeddings = nn.Embedding.from_pretrained(torch.tensor(vocab.embed_matrix, dtype=torch.float32))

    samples = []
    tq_retweets = tqdm(retweets)
    for mid, sample in tq_retweets:
        ids = [mid]
        user_embeddings = []
        content_embeddings = []
        content_mask = []
        dates = []
        relation = {}
        # 微博文本初始语义表示
        source = origin[mid]
        source_text = source["text_seg"]
        source_tokens = source_text.strip().split(" ")
        source_embeddings, source_mask = pad_sentence(token2embed(source_tokens, vocab, pretrained_embeddings, input_dim),
                                                     max_seq_len)
        content_embeddings.append(source_embeddings)
        content_mask.append(source_mask)
        # 用户特征
        source_user = user2embed(source["user"])
        user_embeddings.append(source_user)
        # 微博发布时间
        dates.append(source["date"])

        # 转发微博特征提取
        for item in sample:
            retweet_text = item["text_seg"]
            retweet_tokens = retweet_text.strip().split(" ")
            retweet_embeddings, retweet_mask = pad_sentence(
                                                token2embed(retweet_tokens, vocab, pretrained_embeddings, input_dim),
                                                max_seq_len)
            content_embeddings.append(retweet_embeddings)
            content_mask.append(retweet_mask)

            retweet_user = user2embed(item["user"])
            user_embeddings.append(retweet_user)

            dates.append(item["date"])
            retweet_id = item["mid"]
            ids.append(retweet_id)

            parent = item["parent"]
            if parent in relation:
                relation[parent].append(retweet_id)
            else:
                relation[parent] = [retweet_id]

        content_embeddings, user_embeddings, content_mask, post_mask = pad_post(torch.stack(content_embeddings),
                                                                                torch.stack(user_embeddings),
                                                                                torch.stack(content_mask),
                                                                                max_post_num)
        adj = construct_relation(ids, relation, max_post_num)
        graph_per_interval, post_mask_per_interval = to_graphs(adj, dates, post_mask,
                                                               interval_num, Tags[mid], tunit, max_post_num)
        samples.append(InputSample(mid, content_embeddings, user_embeddings, adj,
                                   graph_per_interval, content_mask, post_mask_per_interval, label))

    del pretrained_embeddings
    return samples


def save_processed_data(samples, processed_datafolder, seq_len, post_num, interval_num, early):
    data = "data_" + str(seq_len) + "_" + str(post_num) \
           + "_" + "entrophy" + "_" + str(interval_num)
    if early != 0:
        data = data + "_" + str(early)
    processed_datafolder = os.path.join(processed_datafolder, data)
    if not os.path.exists(processed_datafolder):
        os.makedirs(processed_datafolder)

    def save_file(data, folder):
        file = os.path.join(folder, str(data.origin_id) + ".pkl")
        if not os.path.exists(file):
            with open(file, "wb") as f:
                    pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

    for sample in tqdm(samples):
        save_file(sample, processed_datafolder)


def create_samples(cnf):
    """
        处理并获取样本数据
        :param cnf: 包含文件参数，模型参数等
        :return:
        """
    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)
    vocab = Vocabulary(cnf.vocabfile, cnf.input_dim)
    Tags = analyze_tags(cnf)
    para = {
        "max_seq_len": cnf.max_seq_len,
        "max_post_num": cnf.max_post_num,
        "interval": cnf.interval,
        "interval_num": cnf.interval_num,
        "input_dim": cnf.input_dim,
        "tunit": cnf.tunit
    }
    # 调试用
    fake_samples = construct_samples(fake, origin=origin, para=para, vocab=vocab, Tags=Tags, label=1)
    nonfake_samples = construct_samples(nonfake, origin=origin, para=para, vocab=vocab, Tags=Tags, label=0)

    # fake_samples = multi_process(data=fake, nthread=4, func=construct_samples,
    #                              origin=origin,  para=para, vocab=vocab, label=1)
    # nonfake_samples = multi_process(data=nonfake, nthread=4, func=construct_samples,
    #                                 origin=origin, para=para, vocab=vocab, label=0)

    samples = fake_samples + nonfake_samples


    del origin, fake, nonfake, vocab
    return samples

def create_samples_mis(cnf, early):
    """
    处理并获取样本数据
    :param cnf: 包含文件参数，模型参数等
    :return:
    """
    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)
    vocab = Vocabulary(cnf.vocabfile, cnf.input_dim)
    Tags = analyze_tags(cnf)
    para = {
        "max_seq_len": cnf.max_seq_len,
        "max_post_num": cnf.max_post_num,
        "interval": cnf.interval,
        "interval_num": cnf.interval_num,
        "input_dim": cnf.input_dim,
        "tunit": cnf.tunit
    }
    # 调试用
    fake_samples = construct_samples(fake, origin=origin, para=para, vocab=vocab, Tags=Tags, label=1)
    nonfake_samples = construct_samples(nonfake, origin=origin, para=para, vocab=vocab, Tags=Tags, label=0)

    # fake_samples = multi_process(data=fake, nthread=4, func=construct_samples,
    #                              origin=origin,  para=para, vocab=vocab, label=1)
    # nonfake_samples = multi_process(data=nonfake, nthread=4, func=construct_samples,
    #                                 origin=origin, para=para, vocab=vocab, label=0)

    samples = fake_samples + nonfake_samples

    save_processed_data(samples, cnf.processed_data, cnf.max_seq_len, cnf.max_post_num, cnf.interval_num, early)

    del origin, fake, nonfake, vocab, samples