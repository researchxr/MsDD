# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 10:06
# @Author  : Naynix
# @File    : util.py

import os
import json
import pickle
from definition.Vocabulary import Vocabulary
from definition.InputSample import MDInputSample as InputSample
from utils.parallel import multi_process
from torch import nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from operator import itemgetter
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
torch.multiprocessing.set_sharing_strategy('file_system')


def save_model(model, cnf, early, train_rate, version, best_result):
    """
    模型保存
    :param model:
    :param cnf:
    :param early:
    :param train_rate:
    :return:
    """
    f1_1 = round(best_result["f1_1"], 4)
    model_name = f"MD_t({str(train_rate)})_e({str(early)})" \
                 f"_para({str(cnf.interval)}_{str(cnf.interval_num)})_{version}_{str(f1_1)}.pt"
    # model_name = f"MD_t(" + str(train_rate) + ") e(" + str(early) + ") para(" + \
    #              str(cnf.interval) + " " + str(cnf.interval_num) + ").pt"
    model_file = os.path.join(cnf.modelfolder, model_name)
    torch.save(model.state_dict(), model_file)


def load_model(model, cnf, early, train_rate):
    """
    模型加载
    :param model:
    :param cnf:
    :param early:
    :param train_rate:
    :return:
    """
    model_name = "MD_t(" + str(train_rate) + ") e(" + str(early) + ") para(" + \
                 str(cnf.interval) + " " + str(cnf.interval_num) + ").pt"
    model_file = os.path.join(cnf.modelfolder, model_name)
    model.load_state_dict(torch.load(model_file))
    return model


def save_result(result, cnf, train_rate, early, version):
    """
    训练结果保存至csv文件，方便实验结果统计分析
    :param best_result:
    :param cnf:
    :param train_rate:
    :param early:
    :return:
    """
    result_file = cnf.result_file
    t = datetime.now().strftime('%Y-%m-%d')
    remark = t + " " + str(version) + "\n"
    remark += " train_rate: {:.2f} ".format(train_rate)
    if early > 0:
        # result_file += "_" + str(early)
        remark += " early:{:d}\n".format(early)
    else:
        if result["f1_1"] < 0.8:
            return
    remark += " interval:{:d}".format(cnf.interval)
    remark += " interval_num:{:d}\n".format(cnf.interval_num)
    result["remark"] = remark

    df = pd.DataFrame(result, index=[0])
    if os.path.exists(cnf.result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, mode='a', header=True, index=False)


def metric(outputs, labels, type_=1):
    """
    模型评价：准确率，精确率，召回率， f1， AUC， AP
    :param outputs:
    :param labels:
    :param type_:
    :return:
    """
    labels = labels.cpu().numpy()
    outputs = outputs.detach()
    outputs_ = np.argmax(outputs.cpu().numpy(), axis=1)
    prec_1, recall_1, f1_1, _ = metrics.precision_recall_fscore_support(labels, outputs_, average="macro",
                                                                  labels=[type_])
    prec_0, recall_0, f1_0, _ = metrics.precision_recall_fscore_support(labels, outputs_, average="macro",
                                                                  labels=[1-type_])
    acc = metrics.accuracy_score(labels, outputs_)

    outputs = F.softmax(outputs, dim=1)
    outputs_ = outputs.cpu().numpy()[:, 1]
    auc = metrics.roc_auc_score(labels, outputs_, average="macro")
    ap = metrics.average_precision_score(labels, outputs_)

    result = {
        "acc": acc,
        "prec_1": prec_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "prec_0": prec_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "auc": auc,
        "ap": ap
    }
    return result


def read_origin(origin_folder):
    """
    读取原微博相关信息，对应 pheme/origin/
    :param origin_folder: 原微博数据文件所在地址
    :return:
    """
    origin = {}
    files = os.listdir(origin_folder)
    for file in files:
        file_path = os.path.join(origin_folder, file)
        with open(file_path, 'r') as f:
            info = json.load(f)

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
            content["text"] = info["text"]
            content["text_seg"] = info["text_seg"]
            content["date"] = info["date"]
            content["user"] = user

            origin[info["mid"]] = content
    return origin


def read_retweets(retweet_folder):
    """
    读取原微博的转发微博相关信息，对应 pheme/fake, pheme/nonfake
    :param retweet_folder:转发微博数据文件所在地址
    :return:
    """
    retweet = []
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
                # content["text"] = info["text"]
                content["date"] = info["date"]
                content["user"] = user
                if info["parent"] == "":
                    content["parent"] = mid
                else:
                    content["parent"] = info["parent"]
                contents.append(content)
            retweet.append((mid, contents))

    return retweet


def time_interval(start, end):
    start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    seconds = (end - start).total_seconds()
    return seconds

def user2embed(user_info):

    name_length = len(user_info["name"])
    if user_info["verified"] == True:
        verified = 1
        verified_type = user_info["verified_type"]
        verified_reason = len(user_info["verified_reason"])
    else:
        verified = 0
        verified_type = 0
        verified_reason = 0
    if user_info["gender"] == "m":
        gender = 0
    else:
        gender = 1
    if user_info["description"] == None:
        desc_length = 0
    else:
        desc_length = len(user_info["description"])
    follower_cnt = user_info["followers_count"]
    followee_cnt = user_info["followees_count"]
    follow_ratio = followee_cnt / (follower_cnt + 1)
    is_influential = 1 if follower_cnt > 1000 else 0

    embeddings = [name_length, verified, verified_type, verified_reason,
                  gender, desc_length, follow_ratio, is_influential]
    return torch.tensor(embeddings, dtype=torch.float32)


def token2embed(tokens, vocab, pretrained_embeddings, input_dim):
    """
    将 一段文本 表示为 初始嵌入矩阵
    :param tokens: 分词处理后的 word list
    :param vocab: 包含 预训练词向量相关信息 的词汇类
    :param pretrained_embeddings: 预训练词向量
    :param input_dim:预训练词向量的嵌入表示维度
    :return:
    """
    processed_tokens = []
    for token in tokens:
        if token not in vocab.word2index:
            processed_tokens.append("<unk>")
        else:
            processed_tokens.append(token)
    index = torch.tensor(itemgetter(*processed_tokens)(vocab.word2index), dtype=torch.long)
    embeddings = pretrained_embeddings(index)
    return embeddings.view(-1, input_dim)


def pad_sentence(embeddings, max_seq_len):
    """
    根据 max_seq_len 扩展 初始语义嵌入矩阵 shape:(XXX,200)-> (max_seq_len,200)
    :param embeddings: 一段文本的嵌入表示
    :param max_seq_len: 最大序列长度
    :return: 嵌入表示，mask矩阵
    """
    if embeddings.shape[0] > max_seq_len:
        embeddings = embeddings[:max_seq_len]
        mask = torch.ones(max_seq_len, dtype=torch.float32)
        return embeddings, mask
    else:
        paddings = torch.zeros((max_seq_len-embeddings.shape[0],embeddings.shape[1]),
                               dtype=torch.float32)
        mask = torch.zeros(max_seq_len,dtype=torch.float32)
        mask[:embeddings.shape[0]] = 1
        embeddings = torch.cat([embeddings, paddings], dim=0)
        return embeddings, mask


def pad_post(content_embeddings, user_embeddings, content_mask, max_post_num):
    """
    根据 max_post_num 扩展内容矩阵，相应的掩码矩阵，用户表示矩阵
    (XXX,100,200)->(max_post_num,100,200)
    :param content_embeddings: 内容表示
    :param user_embeddings: 用户表示
    :param content_mask: 内容掩码
    :param max_post_num: 最大
    :return:
    """
    if content_embeddings.shape[0] > max_post_num:
        content_embeddings = content_embeddings[:max_post_num]
        user_embeddings = user_embeddings[:max_post_num]
        content_mask = content_mask[:max_post_num]
        post_mask = torch.ones(max_post_num, dtype=torch.float32)
        return content_embeddings, user_embeddings, content_mask, post_mask
    else:
        content_paddings = torch.zeros(
            (max_post_num-content_embeddings.shape[0], content_embeddings.shape[1], content_embeddings.shape[2]),
            dtype=torch.float32)
        user_paddings = torch.zeros((max_post_num-user_embeddings.shape[0], user_embeddings.shape[1]),
                                    dtype=torch.float32)
        mask_paddings = torch.zeros((max_post_num-content_mask.shape[0], content_mask.shape[1]),
                                    dtype=torch.float32)
        post_mask = torch.torch.zeros(max_post_num, dtype=torch.float32)
        post_mask[:content_embeddings.shape[0]] = 1

        content_embeddings = torch.cat([content_embeddings, content_paddings], dim=0)
        user_embeddings = torch.cat([user_embeddings, user_paddings], dim=0)
        content_mask = torch.cat([content_mask, mask_paddings], dim=0)
        return content_embeddings, user_embeddings, content_mask, post_mask


def to_graphs(adj, dates, post_mask, interval, interval_num, max_post_num):
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
    for i in range(interval_num-1):
        mask = torch.tensor(time_intervals < interval * (i+1), dtype=torch.float32)
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
    return torch.stack(graph_per_interval), torch.stack(post_mask_per_interval)


def construct_relation(ids, relation, max_post_num):
    """
    根据亲属关系构建表示关系的邻接矩阵
    :param id_map: 由 mid 映射到 序号
    :param relation: 关系字典，类似于邻接表模式
    :param max_post_num:
    :return:
    """
    id_map = {j: i for i, j in enumerate(ids)}
    post_num = max_post_num if max_post_num > len(id_map) else len(id_map)
    adj = torch.zeros((post_num, post_num), dtype=torch.float32)

    for parent, children in relation.items():
        row = torch.tensor(id_map[parent], dtype=torch.long).repeat(len(children))
        col = torch.tensor(itemgetter(*children)(id_map), dtype=torch.long)
        adj[row, col] = 1

    adj = adj[:max_post_num, :max_post_num]
    adj = adj.transpose(1, 0) + torch.eye(max_post_num, dtype=torch.float32)
    return adj


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
    label = kwargs["label"]

    max_seq_len = para["max_seq_len"]
    max_post_num = para["max_post_num"]
    interval = para["interval"]
    interval_num = para["interval_num"]
    input_dim = para["input_dim"]

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
        graph_per_interval, post_mask_per_interval = to_graphs(adj, dates, post_mask, interval,
                                                               interval_num, max_post_num)
        samples.append(InputSample(mid, content_embeddings, user_embeddings, adj,
                                   graph_per_interval, content_mask, post_mask_per_interval, label))

    del pretrained_embeddings
    return samples


def save_processed_data(samples, processed_datafolder, seq_len, post_num, interval, interval_num):
    data = "data_" + str(seq_len) + "_" + str(post_num) \
           + "_" + str(interval) + "_" + str(interval_num)
    processed_datafolder = os.path.join(processed_datafolder, data)
    if not os.path.exists(processed_datafolder):
        os.makedirs(processed_datafolder)

    def save_file(data, folder):
        file = os.path.join(folder, str(data.origin_id) + ".pkl")
        if not os.path.exists(file):
            with open(file, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    para = {
        "max_seq_len": cnf.max_seq_len,
        "max_post_num": cnf.max_post_num,
        "interval": cnf.interval,
        "interval_num": cnf.interval_num,
        "input_dim": cnf.input_dim
    }
    # 调试用
    fake_samples = construct_samples(fake, origin=origin, para=para, vocab=vocab, label=1)
    nonfake_samples = construct_samples(nonfake, origin=origin, para=para, vocab=vocab, label=0)

    # fake_samples = multi_process(data=fake, nthread=4, func=construct_samples,
    #                              origin=origin,  para=para, vocab=vocab, label=1)
    # nonfake_samples = multi_process(data=nonfake, nthread=4, func=construct_samples,
    #                                 origin=origin, para=para, vocab=vocab, label=0)

    samples = fake_samples + nonfake_samples

    del origin, fake, nonfake, vocab

    return samples

def create_samples_mis(cnf):
    """
    处理并获取样本数据
    :param cnf: 包含文件参数，模型参数等
    :return:
    """
    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)
    vocab = Vocabulary(cnf.vocabfile, cnf.input_dim)
    para = {
        "max_seq_len": cnf.max_seq_len,
        "max_post_num": cnf.max_post_num,
        "interval": cnf.interval,
        "interval_num": cnf.interval_num,
        "input_dim": cnf.input_dim
    }
    # 调试用
    fake_samples = construct_samples(fake, origin=origin, para=para, vocab=vocab, label=1)
    nonfake_samples = construct_samples(nonfake, origin=origin, para=para, vocab=vocab, label=0)

    # fake_samples = multi_process(data=fake, nthread=4, func=construct_samples,
    #                              origin=origin,  para=para, vocab=vocab, label=1)
    # nonfake_samples = multi_process(data=nonfake, nthread=4, func=construct_samples,
    #                                 origin=origin, para=para, vocab=vocab, label=0)

    samples = fake_samples + nonfake_samples

    save_processed_data(samples, cnf.processed_data, cnf.max_seq_len, cnf.max_post_num, cnf.interval, cnf.interval_num)

    del origin, fake, nonfake, vocab, samples