# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 20:40
# @Author  : Naynix
# @File    : InputSample.py
class MDInputSample(object):
    def __init__(self, mid, content_embeddings, user_embeddings, adj,
                 graph_per_interval, content_mask, post_mask_per_interval, label):
        self.origin_id = mid
        self.content_embeddings = content_embeddings
        self.user_embeddings = user_embeddings
        self.adj = adj
        self.graph_per_interval = graph_per_interval
        self.content_mask = content_mask
        self.post_mask_per_interval = post_mask_per_interval
        self.label = label


class MDInputSample_v1(object):

    def __init__(self, mid, content_embeddings, user_embeddings, content_mask, post_masks, graphs, label):
        self.origin_id = mid
        self.content_embeddings = content_embeddings
        self.user_embeddings = user_embeddings
        self.content_mask = content_mask
        self.post_masks = post_masks
        self.graphs = graphs
        self.label = label

class PPInputSample(object):
    def __init__(self, mid, content_embeddings, MD_embeddings, prop_graph, post_mask,
                 content_mask, uids, early_states, m_pop_final, final_states):
        self.origin_id = mid
        self.content_embeddings = content_embeddings
        self.MD_embeddings = MD_embeddings
        self.prop_graph = prop_graph
        self.post_mask = post_mask
        self.content_mask = content_mask
        self.uids = uids
        self.m_pop_final = m_pop_final
        self.early_states = early_states
        self.final_states = final_states

