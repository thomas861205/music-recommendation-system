import os
import csv
import random
import numpy as np
import collections
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class MusicDataset(Dataset):
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (mode, batchsz, n_way, k_shot, k_query, resize))

        self.feature = {}
        temp_feature = np.load('../latent/wavenet/{}_feature.npy'.format(mode))
        offset = temp_feature.shape[0] // 5
        assert offset == 200
        for i in range(5):
            self.feature[i] = temp_feature[i*offset:(i+1)*offset]

        self.label = np.load('../latent/wavenet/{}_label.npy'.format(mode))

        self.positive_pool = []
        self.negative_pool = []
        for i in range(5):
            class_set = {0, 1, 2, 3, 4}
            class_set.remove(i)

            self.positive_pool.append(self.feature[i])
            self.negative_pool.append([])
            for j in class_set:
                self.negative_pool[-1].extend(self.feature[j])
            self.negative_pool[-1] = np.array(self.negative_pool[-1])

    def __len__(self):
        return self.batchsz

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        chosen_class = np.random.choice(5, 1)[0]
        positive_pool = self.positive_pool[chosen_class]
        negative_pool = self.negative_pool[chosen_class]

        selected_positive_idx = np.random.choice(positive_pool.shape[0], self.k_shot + self.k_query, False)
        selected_negative_idx = np.random.choice(negative_pool.shape[0], self.k_shot + self.k_query, False)

        support_x = np.concatenate((positive_pool[selected_positive_idx[:self.k_shot]], negative_pool[selected_negative_idx[:self.k_shot]]), axis=0)
        support_y = np.zeros((self.setsz), dtype=np.int)
        support_y[:self.k_shot] = 1

        query_x = np.concatenate((positive_pool[selected_positive_idx[self.k_shot:]], negative_pool[selected_negative_idx[self.k_shot:]]), axis=0)
        query_y = np.zeros((self.querysz), dtype=np.int)
        query_y[:self.k_query] = 1

        support_x = torch.from_numpy(support_x).float()
        support_y = torch.from_numpy(support_y).long()

        query_x = torch.from_numpy(query_x).float()
        query_y = torch.from_numpy(query_y).long()

        return support_x.permute(0, 2, 1).unsqueeze(-1), support_y, query_x.permute(0, 2, 1).unsqueeze(-1), query_y


if __name__ == '__main__':
    import time
    from matplotlib import pyplot as plt

    music = MusicDataset(root='', mode='train', n_way=2, k_shot=5, k_query=5, batchsz=1000, resize=-1)

    for i, set_ in enumerate(music):
        support_x, support_y, query_x, query_y = set_
        print(support_x.shape)
        print(support_y.shape)

        print(query_x.shape)
        print(query_y.shape)
        exit()
