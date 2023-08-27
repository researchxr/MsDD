# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 11:02
# @Author  : Naynix
# @File    : MD_Dataset.py

from dgl.data import DGLDataset
from torch.utils.data import Dataset
import os
import pickle
import torch

class MD_Dataset(Dataset):
    def __init__(self, datafolder, origin_ids):
        self.processed_data = datafolder
        self.ids = origin_ids

    def __getitem__(self, index):
        file = os.path.join(self.processed_data, self.ids[index] + ".pkl")
        with open(file, "rb") as f:
            line = pickle.load(f)

        label = torch.tensor(line.label)
        return [line.content_embeddings, line.user_embeddings, line.adj, line.graph_per_interval,
                line.content_mask, line.post_mask_per_interval, label]

    def __len__(self):
        return len(self.ids)

class MD_Dataset_para(Dataset):
    def __init__(self, datafolder, origin_ids):
        self.processed_data = datafolder
        self.ids = origin_ids
        self.id2order = {str(id): i for i, id in enumerate(self.ids)}
        self.order2id = {i: str(id) for i, id in enumerate(self.ids)}

    def __getitem__(self, index):
        file = os.path.join(self.processed_data, self.ids[index] + ".pkl")
        order = self.id2order[self.ids[index]]
        with open(file, "rb") as f:
            line = pickle.load(f)

        label = torch.tensor(line.label)
        return [line.content_embeddings, line.user_embeddings, line.adj, line.graph_per_interval,
                line.content_mask, line.post_mask_per_interval, label, order]

    def __len__(self):
        return len(self.ids)