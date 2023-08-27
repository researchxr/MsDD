# -*- coding: utf-8 -*-
# @Time    : 2022/1/14 20:56
# @Author  : Naynix
# @File    : loss_curve.py

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
folder = "/mntc/yxy/MDPP/output/logs/remote"
dataset = "misinfdect"

file = "loss_main_v0.pkl"

file_path = os.path.join(folder, dataset, file)

def plot_loss():
    with open(file_path, "rb") as f:
        losses = pickle.load(f)
    losses = np.array(losses).reshape(-1, 50)
    losses = np.mean(losses,axis=1)
    x = [i for i in range(len(losses))]

    plt.plot(x, losses, color="red", linewidth=1)
    plt.show()
    print("ok")
plot_loss()