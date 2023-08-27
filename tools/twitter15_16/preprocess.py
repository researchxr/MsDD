# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 10:46
# @Author  : Naynix
# @File    : preprocess.py
import torch
import os
import pickle
import json
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import queue

stemmer = SnowballStemmer("english")


def normalizeToken(token):
    lowercased_token = token.lower()
    order = re.compile(r'^\d+th$|^\d+st$|^\d+nd$|^\d+rd$')
    emoj = re.compile(r'^:\w+:$')
    number = re.compile(r'\d+')
    if token.startswith("@"):
        # return "@USER"
        return '<user>'
    elif token.startswith("#"):
        return '<hashtag>'
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        # return "HTTPURL"
        return '<url>'
    elif order.match(token) or number.match(token):
            return '<number>'
    elif emoj.match(token):
        return 'emoji'
    elif len(token) == 1:
        emo = demojize(token)
        if emoj.match(emo):
            return 'emoji'
        else :
            return emo
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return ""
        elif token == '...' or token == '..' or token == '.':
            return ""
        else:
            return re.split(r'[;,\.\-\s]\s*', token)


def normalizeTweet(tweet):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    new_tokens = []
    for token in tokens:

            token = normalizeToken(token)
            if isinstance(token,list):
                for item in token:
                    new_tokens.append(item)
            elif len(token)==0:
                pass
            else:
                new_tokens.append(token)
    normTweet = " ".join([token for token in new_tokens])

    normTweet = normTweet.replace("can't ", "cannot ").replace("n't ", " n't ").replace("n 't ", " n't ").replace(
        "ca n't", "can't").replace("don't ", "donot ").replace("aren't ", "arenot ")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ",
                                                                                                         " 'll ").replace(
        "'d ", " 'd ").replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.").replace(" p . m ", " p.m ").replace(" a . m .",
                                                                                            " a.m.").replace(" a . m ",
                                                                                                             " a.m ")

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})-([0-9]{2,4})", r"\1-\2", normTweet)

    return normTweet.split()


def text_split(text):
    stopworddic = set(stopwords.words('english'))
    res = text.split(' ')
    res = [i.lower() for i in res if i not in stopworddic]
    res = ' '.join(res)
    res = normalizeTweet(res)
    return " ".join(res)

def read_twitter(file, vocab_size):
    samples = {}
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            mid = line[0]
            parent_id = line[1]
            number = line[2]
            repr = line[-1]

            repr = repr.strip().split(" ")

            indices = []
            vals = []
            for item in repr:
                index, val = item.strip().split(":")
                indices.append(int(index))
                vals.append(int(val))
            vec = torch.zeros(vocab_size, dtype=torch.float)
            indices = torch.tensor(indices)
            vals = torch.tensor(vals, dtype=torch.float)
            vec[indices] = vals
            item = (parent_id, number, vec)
            if mid in samples:
                samples[mid].append(item)
            else:
                samples[mid] = [item]
    return samples


def read_labels(datafolder, dataset):
    label_file = os.path.join(datafolder, "label.txt")
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

def check_tree(rel, uids):
    q = queue.Queue()
    q.put(rel["ROOT"][0])
    node_ids = set()
    new_rel = {}
    delay = {}
    while not q.empty():
        cur = q.get()
        node_ids.add(cur)
        delay[cur] = uids[cur]
        if cur not in rel:
            continue
        for child in rel[cur]:
            if child in node_ids:
                continue
            q.put(child)
            if cur in new_rel:
                new_rel[cur].append(child)
            else:
                new_rel[cur] = [child]
    return new_rel, delay

def read_tree(treefolder):

    pattern = re.compile(r'\'(.*?)\'')  # 查找数字

    Tree = {}
    files = os.listdir(treefolder)
    for file in files:
        mid = file.strip().split(".")[0]
        filepath = os.path.join(treefolder, file)
        with open(filepath, "r") as f:
            uids = {}
            lines = f.readlines()
            rel = {}
            for line in lines:
                items = pattern.findall(line)
                parent = items[0]
                child = items[3]
                delay = float(items[-1])
                uids[child] = delay

                if parent not in rel:
                    rel[parent] = [child]
                else:
                    rel[parent].append(child)

            rel_aftercheck, time_delay = check_tree(rel, uids)
            dic = {}
            dic["rel"] = rel_aftercheck
            dic["time_delay"] = time_delay
            Tree[mid] = dic
    return Tree


def read_source(source_file):
    source = {}
    with open(source_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            dic = {}
            id, text = line.strip().split("\t")
            text_seg = text_split(text)
            dic["text"] = text
            dic["text_seg"] = text_seg
            source[id] = dic
    return source


if __name__ == "__main__":
    dataset = "twitter16"
    datafolder = "/mntc/yxy/MDPP/data"
    datafolder = os.path.join(datafolder, dataset)

    # labels = read_labels(datafolder, dataset)

    treefolder = os.path.join(datafolder, "tree")
    Tree = read_tree(treefolder)
    tree_file = os.path.join(datafolder, "tree.pkl")

    source_file = os.path.join(datafolder, "source_tweets.txt")
    source = read_source(source_file)
    new_source_file = os.path.join(datafolder, "source_tweets.json")


    with open(tree_file, "wb") as f:
        pickle.dump(Tree, f)

    with open(new_source_file, "w") as f:
        json.dump(source, f, ensure_ascii=False, indent=True)








