# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 15:03
# @Author  : Naynix
# @File    : data_analysis.py


"""
第二个任务之前的数据统计分析

"""
from utils.util import read_origin, read_retweets, time_interval
from config.config import Config
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import math

parser = argparse.ArgumentParser(description="default")
parser.add_argument("--dataset", type=str, help="required", default="new_pheme")
args = parser.parse_args()

cnf = Config(dataset=args.dataset, version="")


def CDF(datasets, names, xlabel, xticks):
    cur = 0
    for name, data in zip(names, datasets):
        cur += 1
        data.sort()
        plotDataset = [[], []]
        count = len(data)
        val = 1
        point = tuple()
        for i in range(count):
            plotDataset[0].append(data[i])
            plotDataset[1].append((i + 1) / count)
            if math.fabs((i+1) / count - 0.95) < val:
                point = (round(data[i], 4), round((i+1) / count, 4))
                val = math.fabs((i+1) / count - 0.95)

        plt.plot(plotDataset[0], plotDataset[1], '-', linewidth=1, label=name)

        line = [0.95 for _ in range(len(data))]
        plt.plot(plotDataset[0], line, '-', linewidth=1, color='red')
        plt.annotate(str(point), xy=point, xytext=(point[0], point[1]-0.13*cur),
                     arrowprops=dict(facecolor='k', headwidth=8, width=1))

    plt.legend()
    plt.xticks(xticks[0], xticks[1])
    plt.xlabel(xlabel)
    plt.ylabel("P")
    plt.show()


def total_time(Retweets, origin, h_limit):
    times = []
    for mid, retweets in Retweets:
        start = origin[mid]["date"]
        span = 0
        for retweet in retweets:
            t = retweet["date"]
            t_span = time_interval(start, t)/3600.0
            if t_span > span:
                span = t_span
        if span < h_limit:
            times.append(span)
    return np.array(times)


def total_amount(Retweets, amount_limit):
    amounts = []
    for mid, retweets in Retweets:
        if amount_limit:
            if len(retweets) + 1 < amount_limit:
                amounts.append(len(retweets) + 1)
        else:
            amounts.append(len(retweets) + 1)
    return np.array(amounts)


def time_analysis(origin, fake):
    """
    时间累积分布情况
    :param origin:
    :param fake:
    :return:
    """
    h_limit = 2400
    fake_times = total_time(fake, origin, h_limit)
    xticks = [[i for i in range(0, h_limit, int(h_limit/5))],
              [str(i) + "h" for i in range(0, h_limit, int(h_limit/5))]]
    CDF([fake_times], ["fake"], "Time", xticks)


def amount_analysis(fake):
    """
    数量累积分布情况
    :param fake:
    :return:
    """
    amount_limit = 256
    fake_amounts = total_amount(fake, amount_limit)
    xticks = [[i for i in range(0, amount_limit, int(amount_limit/5))],
              [str(i) + "" for i in range(0, amount_limit, int(amount_limit/5))]]
    CDF([fake_amounts], ["fake"], "Amount", xticks)


def time_amount_analysis(origin, Retweets, h_limit):
    """
    一定时间段内的数量累积分布情况
    :param origin:
    :param Retweets:
    :param h_limit:
    :return:
    """
    fake_amounts = []
    for mid, retweets in Retweets:
        start = origin[mid]["date"]
        num = 0
        for retweet in retweets:
            t = retweet["date"]
            t_span = time_interval(start, t) / 3600.0
            if t_span < h_limit:
                num += 1
        fake_amounts.append(num)

    amount_limit = max(fake_amounts)
    xticks = [[i for i in range(0, amount_limit, int(amount_limit / 5))],
              [str(i) + "" for i in range(0, amount_limit, int(amount_limit / 5))]]
    CDF([fake_amounts], ["fake"], "Time" + str(h_limit) + "-Amount", xticks)


def amount_interval_analysis(origin, Retweets, amount_limit,
                             h_limit_1):
    """
    数量在不同时间段的分布情况
    :param origin:
    :param Retweets:
    :param h_limit:
    :return:
    """
    X = []
    Y = []
    for mid, retweets in Retweets:
        start = origin[mid]["date"]
        num_1 = 0
        num_2 = 0
        for retweet in retweets:
            t = retweet["date"]
            t_span = time_interval(start, t) /3600.0
            if t_span < h_limit_1:
                num_1 += 1
            if t_span > h_limit_1:
                num_2 += 1
        X.append(num_1)
        Y.append(num_2)

    point_X= np.array(X)
    point_Y = np.array(Y)

    X = point_X[point_X < amount_limit]
    Y = point_Y[point_X < amount_limit]
    plt.scatter(x=X, y=Y,s=1, c='black')

    plt.xlabel("nums of tweet in 0 ~ %dh" % h_limit_1)
    plt.ylabel("nums of tweet in %d ~ end" % h_limit_1)
    plt.show()

    amount_limit_X = max(X)
    amount_limit_Y = max(Y)
    amount_limit = max(amount_limit_X, amount_limit_Y)
    xticks = [[i for i in range(0, amount_limit, int(amount_limit / 5))],
              [str(i) + "" for i in range(0, amount_limit, int(amount_limit / 5))]]
    CDF([list(X), list(Y)],
        ["0 ~ %dh" % h_limit_1, "%d ~ end" % h_limit_1],
        "Amount-Interval", xticks)




def collect_times(origin, Retweets):

    Spans = []
    for mid, retweets in Retweets:
        start = origin[mid]["date"]
        for retweet in retweets:
            t = retweet["date"]
            t_span = time_interval(start, t) / 3600.0
            Spans.append([mid, t_span])
    return Spans


def time_distribution(origin, fake, h_limit):
    """
    各转发推文的时间分布情况
    :param origin:
    :param fake:
    :param h_limit:
    :return:
    """
    fake_spans = collect_times(origin, fake)
    spans = fake_spans
    mids = set([item[0] for item in spans])
    mid_map = {j: i for i, j in enumerate(mids)}

    spans = [[mid_map[item[0]], item[1]] for item in spans]
    spans = np.array(spans)
    spans = spans[spans[:, 1] < h_limit]
    X = spans[:, 0]
    Y = spans[:, 1]
    plt.scatter(x=X, y=Y, s=1, c='black')

    lines_2h = [[mid_map[item], 1] for item in mids]
    lines_2h = np.array(lines_2h)

    lines_3h = [[mid_map[item], 3] for item in mids]
    lines_3h = np.array(lines_3h)

    lines_4h = [[mid_map[item], 5] for item in mids]
    lines_4h = np.array(lines_4h)

    plt.scatter(x=lines_2h[:, 0], y=lines_2h[:, 1], s=1, c='red')
    plt.scatter(x=lines_3h[:, 0], y=lines_3h[:, 1], s=1, c='green')
    plt.scatter(x=lines_4h[:, 0], y=lines_4h[:, 1], s=1, c='orange')

    plt.xlabel("Number of Propagation Tree" )
    plt.ylabel("Time")

    plt.show()

def user_analysis(origin, Retweets):
    """
    一定时间段，参与用户的数量，以及用户发布推文数量的经验累积分布
    :param origin:
    :param Retweets:
    :param h_limit:
    :return:
    """
    users = {}
    tweets = 0
    for mid, retweets in Retweets:
        tweets += 1
        if origin[mid]["user"]["_id"] in users:
            users[origin[mid]["user"]["_id"]] += 1
        else:
            users[origin[mid]["user"]["_id"]] = 1
        for retweet in retweets:
            tweets += 1
            if retweet["user"]["_id"] in users:
                users[retweet["user"]["_id"]] += 1
            else:
                users[retweet["user"]["_id"]] = 1

    print(tweets, len(users))
    numofuser = []
    for num in users.values():
        numofuser.append(num)

    user_limit = max(numofuser)
    xticks = [[i for i in range(0, user_limit, int(user_limit / 5))],
              [str(i) + "" for i in range(0, user_limit, int(user_limit / 5))]]
    CDF([numofuser],
        ["Amount of Tweets"],
        "Amount of Tweets", xticks)


def analysis():
    origin = read_origin(cnf.originfolder)
    fake = read_retweets(cnf.fakefolder)
    nonfake = read_retweets(cnf.nonfakefolder)
    # time_analysis(origin, fake) # 各传播树 时间跨度（最后一条推文的时间 - 原推文发布时间） 经验累积分布
    # amount_analysis(fake) #  各传播树 节点数量（推文数量） 经验累积分布
    time_distribution(origin, fake, 24) #  各传播树中 推文发布时间分布情况

    # time_amount_analysis(origin, fake,  24)  # 时间限制下 各传播树 节点数量的经验累积分布
    # amount_interval_analysis(origin, fake, 2048, 12)  # 提供 0~a, a~end 两个时间段的 传播书中 节点数量的经验累积分布

    # user_analysis(origin, nonfake)


# analysis()