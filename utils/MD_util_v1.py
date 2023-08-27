# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 10:28
# @Author  : Naynix
# @File    : MD_util_v1.py
from utils.util import read_origin, read_retweets, pad_sentence, token2embed, user2embed, pad_post, time_interval
from definition.Vocabulary import Vocabulary
from utils.parallel import multi_process
import torch
from torch import nn
from tqdm import tqdm
import dgl
from definition.InputSample import MDInputSample_v1 as InputSample


def construct_graphs(ids, relation, dates, interval, interval_num, max_post_num):
    idmap = {j: i for i, j in enumerate(ids)}
    start = dates[0]
    delays = [time_interval(start, date) for date in dates]
    graphs = []
    post_masks = []
    for i in range(interval_num):
        time_limit = interval * (i+1)
        edges_from = []
        edges_to = []
        nodes = set()
        for parent, children in relation.items():
            p_id = idmap[parent]
            if delays[p_id] < time_limit and p_id < max_post_num:
                nodes.add(p_id)
                for child in children:
                    c_id = idmap[child]
                    if delays[c_id] < time_limit and c_id < max_post_num:
                        edges_from.append(p_id)
                        edges_to.append(c_id)
                        nodes.add(c_id)
        post_mask = torch.zeros(max_post_num, dtype=torch.float)
        nodes = torch.tensor([list(nodes)], dtype=torch.long)
        post_mask[nodes] = 1
        if len(edges_from) == 0:
            graph = dgl.DGLGraph()
            graph.add_nodes(num=max_post_num)
        else:
            edges_from, edges_to = torch.tensor(edges_from), torch.tensor(edges_to)
            graph = dgl.graph((edges_from, edges_to), num_nodes=max_post_num)
        graphs.append(graph)
        post_masks.append(post_mask)
    graphs = dgl.batch(graphs)
    post_masks = torch.stack(post_masks)
    return graphs, post_masks

def construct_samples(Retweets, **kwargs):
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
    tq_retweets = tqdm(Retweets)
    for mid, retweets in tq_retweets:
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
        source_embeddings, source_mask = pad_sentence(
            token2embed(source_tokens, vocab, pretrained_embeddings, input_dim),
            max_seq_len)
        content_embeddings.append(source_embeddings)
        content_mask.append(source_mask)
        # 用户特征
        source_user = user2embed(source["user"])
        user_embeddings.append(source_user)
        # 微博发布时间
        dates.append(source["date"])

        # 转发微博特征提取
        for retweet in retweets:
            retweet_text = retweet["text_seg"]
            retweet_tokens = retweet_text.strip().split(" ")
            retweet_embeddings, retweet_mask = pad_sentence(
                token2embed(retweet_tokens, vocab, pretrained_embeddings, input_dim),
                max_seq_len)
            content_embeddings.append(retweet_embeddings)
            content_mask.append(retweet_mask)

            retweet_user = user2embed(retweet["user"])
            user_embeddings.append(retweet_user)

            dates.append(retweet["date"])
            retweet_id = retweet["mid"]
            ids.append(retweet_id)

            parent = retweet["parent"]
            if parent in relation:
                relation[parent].append(retweet_id)
            else:
                relation[parent] = [retweet_id]

        content_embeddings, user_embeddings, content_mask, post_mask = pad_post(torch.stack(content_embeddings),
                                                                                torch.stack(user_embeddings),
                                                                                torch.stack(content_mask),
                                                                                max_post_num)

        graphs, post_masks = construct_graphs(ids, relation,dates, interval, interval_num, max_post_num)

        samples.append(InputSample(mid, content_embeddings, user_embeddings,
                                   content_mask, post_masks, graphs, label))

    del pretrained_embeddings
    return samples


def create_samples_v1(cnf):
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