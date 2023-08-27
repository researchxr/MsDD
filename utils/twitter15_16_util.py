# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 21:23
# @Author  : Naynix
# @File    : twitter15_16_util.py
import pickle
import json
from definition.InputSample import MDInputSample as InputSample
from definition.Vocabulary import Vocabulary
from utils.util import token2embed, pad_sentence, construct_relation
import torch
from torch import nn
import numpy as np
from tqdm import tqdm


def read_labels(label_file):
    label_dic = {
        "true": 1,
        "false": 0,
        "unverified": 1,
        "non-rumor": 1
    }
    labels = {}
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            label, id = line.strip().split(":")
            labels[id] = label_dic[label]

    return labels


def read_tree(tree_file):
    with open(tree_file,"rb") as f:
        tree = pickle.load(f)

    return tree


def read_source(source_file):
    with open(source_file, "r") as f:
        source = json.load(f)
    return source

def to_graphs(adj, delay, post_mask, interval,interval_num, max_post_num):

    time_intervals = np.array([t*60 for _, t in delay])
    graph_per_interval = []
    post_mask_per_interval = []
    adj = adj.transpose(1, 0)
    for i in range(interval_num - 1):
        mask = torch.tensor(time_intervals < interval * (i + 1), dtype=torch.float)
        if mask.shape[0] < max_post_num:
            mask = torch.cat([mask, torch.zeros(max_post_num - mask.shape[0], dtype=torch.float)], dim=0)
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

def construct_samples(source, **kwargs):
    tree = kwargs["tree"]
    labels = kwargs["labels"]
    vocab = kwargs["vocab"]
    para = kwargs["para"]

    max_seq_len = para["max_seq_len"]
    max_post_num = para["max_post_num"]
    interval = para["interval"]
    interval_num = para["interval_num"]
    input_dim = para["input_dim"]

    pretrained_embeddings = nn.Embedding.from_pretrained(torch.tensor(vocab.embed_matrix, dtype=torch.float))

    samples = []

    for mid, tweet in tqdm(source.items()):
        source_tokens = tweet["text_seg"].strip().split(" ")

        source_embeddings, source_mask = pad_sentence(token2embed(source_tokens, vocab, pretrained_embeddings, input_dim),
                                         max_seq_len)
        retweet_embeddings = torch.zeros((max_post_num-1, max_seq_len, input_dim), dtype=torch.float)
        retweet_mask = torch.zeros((max_post_num-1, max_seq_len), dtype=torch.long)

        content_embeddings = torch.cat([source_embeddings.view(-1, max_seq_len, input_dim), retweet_embeddings], dim=0)
        content_mask = torch.cat([source_mask.view(-1, max_seq_len), retweet_mask], dim=0)

        user_embeddings = torch.zeros((max_post_num, 8), dtype=torch.float)

        rel = tree[mid]["rel"]
        delay = tree[mid]["time_delay"]
        delay = [[uid, delay] for uid, delay in delay.items()]
        delay = sorted(delay, key=lambda x: x[1])
        uids = [uid for uid, _ in delay]

        post_mask = torch.zeros(max_post_num, dtype=torch.long)
        if len(uids) < max_post_num:
            post_mask[:len(uids)] = 1
        else:
            post_mask[:max_post_num] = 1


        adj = construct_relation(uids, rel, max_post_num)
        graph_per_interval, post_mask_per_interval = to_graphs(adj, delay, post_mask, interval,
                                                               interval_num, max_post_num)
        samples.append(InputSample(mid, content_embeddings, user_embeddings, adj,
                                   graph_per_interval, content_mask, post_mask_per_interval, labels[mid]))
    return samples



def T_create_samples(cnf):
    source = read_source(cnf.source_file)
    tree = read_tree(cnf.tree_file)
    labels = read_labels(cnf.label_file)
    para = {
        "max_seq_len": cnf.max_seq_len,
        "max_post_num": cnf.max_post_num,
        "interval": cnf.interval,
        "interval_num": cnf.interval_num,
        "input_dim": cnf.input_dim
    }
    vocab = Vocabulary(cnf.vocabfile, cnf.input_dim)
    samples = construct_samples(source, tree=tree, labels=labels, vocab=vocab, para=para)
    return samples

