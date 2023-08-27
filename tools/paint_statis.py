# -*- coding: utf-8 -*-
# @Time    : 2022/2/15 16:20
# @Author  : Naynix
# @File    : paint_statis.py
from utils.util import read_origin, read_retweets, time_interval
from config.config import Config
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
# 查找字体路径
print(matplotlib.matplotlib_fname())
# 查找字体缓存路径
print(matplotlib.get_cachedir())
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
import pandas as pd
import os
from collections import Counter
import seaborn as sns
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

from tools.data_statis import average_tweet_num, average_time_span, get_Tree_info
parser = argparse.ArgumentParser(description="default")
parser.add_argument("--dataset", type=str, help="required", default="new_pheme")
args = parser.parse_args()

cnf = Config(dataset=args.dataset, version="")

def get_text_len(origin , Retweets):
    text_len = []
    for mid, retweets in Retweets:
        text_len.append(len(origin[mid]["text_seg"].split(" ")))
        for retweet in retweets:
            text_len.append(len(retweet["text_seg"].split(" ")))
    return text_len

def Draw_loglog(X, Y, labels, name):
    X = np.array(X)
    Y = np.array(Y)

    fontsize = 16

    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(X, Y,s=10, c='black')
    # ax.set_xticks(fontsize=fontsize)
    # ax.set_yticks(fontsize=fontsize)
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    fig.savefig(f"{args.dataset}-{name}.pdf")

    """
    x_min = np.min(X)
    x_max = np.max(X)
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(X, Y, s=10, c='black')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(labels[0],fontsize=16)
    plt.ylabel(labels[1],fontsize=16)
    plt.xlim(x_min, x_max)
    # plt.show()
    plt.savefig(f"{args.dataset}-{name}.pdf")
    plt.close()
    """

def draw_tweet_num(fake, nonfake):
    f_tweet_num_list,_ = average_tweet_num(fake)
    nf_tweet_num_list, _ = average_tweet_num(nonfake)
    tweet_num_list = f_tweet_num_list + nf_tweet_num_list
    res = Counter(tweet_num_list)
    new_res = [(key, val) for key, val in res.items()]

    new_res = sorted(new_res, key=lambda x: x[0])
    x = [item[0] for item in new_res]
    y = [item[1] for item in new_res]
    labels = ["Number of Retweets", "Number of Origin Tweets"]
    Draw_loglog(x, y, labels, "tweet_num")

def draw_depth(fake, nonfake):
    f_depth_list, _, _,_  = get_Tree_info(fake)
    nf_depth_list, _, _,_ = get_Tree_info(nonfake)
    depth_list = f_depth_list + nf_depth_list
    res = Counter(depth_list)
    new_res = [(key, val) for key, val in res.items()]
    new_res = sorted(new_res, key=lambda x: x[0])
    x = [item[0] for item in new_res]
    y = [item[1] for item in new_res]
    labels = ["Depth of Propagation Tree", "Number of Origin Tweets"]
    Draw_loglog(x, y, labels, "depth")


def draw_text_len(origin, fake, nonfake):
    f_text_len = get_text_len(origin, fake)
    nf_text_len = get_text_len(origin, nonfake)
    text_len_list = f_text_len + nf_text_len
    res = Counter(text_len_list)
    new_res = [(key, val) for key, val in res.items()]
    new_res = sorted(new_res, key=lambda x: x[0])
    x = [item[0] for item in new_res]
    y = [item[1] for item in new_res]
    labels = ["Number of words", "Number of Tweets"]
    Draw_loglog(x, y, labels, "text_len")



def draw_time_span(origin, fake, nonfake):
     f_time_list, _ = average_time_span(origin, fake)
     nf_time_list, _ = average_time_span(origin, nonfake)
     time_list = f_time_list + nf_time_list
     time_list = list(map(int, time_list))

     fontsize = 16

     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
     ax.hist(time_list, bins=100, histtype="stepfilled", alpha=0.3,log=True,color='blue')
     # ax.xticks(fontsize=fontsize)
     # ax.yticks(fontsize=fontsize)
     # ax.set_xticks(fontsize=fontsize)
     # ax.set_yticks(fontsize=fontsize)
     ax.set_yscale("log")
     ax.set_xlabel("Time Range (hour)", fontsize=fontsize)
     ax.set_ylabel("Number of Origin Tweets", fontsize=fontsize)
     fig.savefig(f"{args.dataset}-time_range.pdf")
     """
     # x = stats.gamma(3).rvs(5000)
     # plt.hist(x, bins=80) # 每个bins都有分界线
     # 若想让图形更连续化 (去除中间bins线) 用histtype参数
     plt.hist(time_list, bins=100, histtype="stepfilled", alpha=0.3,log=True,color='blue')
     plt.xticks(fontsize=16)
     plt.yticks(fontsize=16)

     plt.xlabel("Time Range (hour)",fontsize=16)
     plt.ylabel("Number of Origin Tweets",fontsize=16)
     # plt.gca().set_xscale("log")
     plt.savefig(f"{args.dataset}-time_range.pdf")
     # plt.close()

     # ax = sns.distplot(time_list)
     # ax1 = sns.kdeplot(time_list, label='x',
     #                   shade=True, color='r')
     plt.show()
     # plt.show()

     # new_res = [(i,j) for i,j in enumerate(time_list)]
     # X = [item[0] for item in new_res]
     # Y = [int(item[1]/24) for item in new_res]
     # plt.scatter(x=X, y=Y, s=1, c='black')
     # plt.savefig(f"{args.dataset}-time_span")
     # plt.close()


     # time_list = [int(i/24) for i in time_list]
     # res = Counter(time_list)
     # new_res = [(key, val) for key, val in res.items()]
     # new_res = sorted(new_res, key=lambda x: x[0])
     # x = [item[0] for item in new_res]
     # y = [item[1] for item in new_res]
     # labels = ["Time Span", "Number of Events"]
     # Draw_loglog(x, y, labels, "time_span")
    """
if __name__ == "__main__":
    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)
    # draw_tweet_num(fake, nonfake)
    # draw_depth(fake, nonfake)
    # draw_text_len(origin, fake, nonfake)

    draw_time_span(origin, fake, nonfake)