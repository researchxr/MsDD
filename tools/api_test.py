# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 9:28
# @Author  : Naynix
# @File    : api_test.py
import dgl
import torch
edges_from = torch.tensor([1, 2, 3])
edges_to = torch.tensor(([2, 3, 4]))
graph = dgl.graph((edges_from, edges_to))


print(":ok")