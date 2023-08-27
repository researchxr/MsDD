# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 10:12
# @Author  : Naynix
# @File    : Vocabulary.py
class Vocabulary(object):
    def __init__(self, vocabfile, input_dim):
        self.vocabfile = vocabfile
        self.input_dim = input_dim
        self.word2index = {}
        self.index2word = {}
        self.embed_matrix = []

        self.read_vocab()
        self.word2index["<unk>"] = len(self.embed_matrix)
        self.index2word[len(self.embed_matrix)] = "<unk>"
        self.embed_matrix.append([0] * self.input_dim)

    def read_vocab(self):
        with open(self.vocabfile, 'r') as f:
            vocab_num, dim = list(map(int, f.readline().strip().split(" ")))
            assert dim == self.input_dim, "the embedding dim is not equal to %d" % self.input_dim
            lines = f.readlines()
            indices = [i for i in range(0, len(lines))]

            for index, line in zip(indices, lines):
                line = line.strip().split(" ")
                word = line[0]
                embedding = list(map(float, line[1:]))

                self.index2word[index] = word
                self.word2index[word] = index
                self.embed_matrix.append(embedding)