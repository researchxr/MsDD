# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 19:42
# @Author  : Naynix
# @File    : process_repr.py
import pickle
import argparse
from config.config import Config
import os

parser = argparse.ArgumentParser(description="Parameters of Propagation Predicted Experiment")
parser.add_argument("--reprfile", type=str, help="required", default="MD_repr(0.8) e(3) para(3600 32).pkl")
parser.add_argument("--dataset", type=str, help="required", default="misinfdect")
args = parser.parse_args()
cnf = Config(dataset=args.dataset)


def process_repr():

    reprfile = os.path.join(cnf.reprfolder,args.reprfile)
    with open(reprfile, "rb") as f:
        repr = pickle.load(f)

    os.remove(reprfile)
    n_repr = {}
    for mid, val in repr.items():
        nval = repr[mid][-1, ::]
        n_repr[mid] = nval

    with open(reprfile, "wb") as f:
        pickle.dump(repr, f)

process_repr()