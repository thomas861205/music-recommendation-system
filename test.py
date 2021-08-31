import os
import pickle
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import *
from utils import *

device = torch.device("cpu")
net = NetLatent()
net.load_state_dict(torch.load("classifier.pth", map_location="cpu"))
net.to(device)
net.eval()
_num_rounds = 20

#####################################################################################################################

# test_latent = np.load('./latent/classifier/train_latent.npy')
# test_label = np.load('./latent/classifier/train_label.npy')
# connected = cal_connected(test_latent)
# W, T = cal_segmentation_default(test_latent, connected)

# num_round=_num_rounds
# chosen_label=np.random.choice([0, 1, 2, 3, 4], 1)[0]
# Q, Y, S = run_segmentation(test_label, W, T, num_round, target_label=chosen_label)

# TP = 0
# FP = 0
# FN = 0
# TN = 0

# S = S.astype(int)
# binary_label = (test_label == chosen_label).astype(int)
# for i in range(test_label.shape[0]):
#     if binary_label[i] == S[i, 0] and binary_label[i] == 1:
#         TP += 1
#     elif S[i, 0] == 1 and binary_label[i] == 0:
#         FP += 1
#     elif S[i, 0] == 0 and binary_label[i] == 1:
#         FN += 1
#     else:
#         TN += 1
# print('TP: {}, FP: {}, FN: {}, TN: {}, Sum: {}'.format(TP, FP, FN, TN, TP + FP + FN + TN))
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# print('Precision: {:.2f}, Recall: {:.2f}, Accuracy: {:.2f}'.format(precision, recall, (TP + TN) / (TP + FP + FN + TN)))

#####################################################################################################################

# test_latent = np.load('./latent/classifier/test_latent.npy')
# test_label = np.load('./latent/classifier/test_label.npy')
# connected = cal_connected(test_latent)
# W, T = cal_segmentation_default(test_latent, connected)

# num_round=_num_rounds
# chosen_label=np.random.choice([5, 6, 7, 8, 9], 1)[0]
# Q, Y, S = run_segmentation(test_label, W, T, num_round, target_label=chosen_label)

# TP = 0
# FP = 0
# FN = 0
# TN = 0

# S = S.astype(int)
# binary_label = (test_label == chosen_label).astype(int)
# for i in range(test_label.shape[0]):
#     if binary_label[i] == S[i, 0] and binary_label[i] == 1:
#         TP += 1
#     elif S[i, 0] == 1 and binary_label[i] == 0:
#         FP += 1
#     elif S[i, 0] == 0 and binary_label[i] == 1:
#         FN += 1
#     else:
#         TN += 1
# print('TP: {}, FP: {}, FN: {}, TN: {}, Sum: {}'.format(TP, FP, FN, TN, TP + FP + FN + TN))
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# print('Precision: {:.2f}, Recall: {:.2f}, Accuracy: {:.2f}'.format(precision, recall, (TP + TN) / (TP + FP + FN + TN)))

#####################################################################################################################

test_latent = np.load('./latent/wavenet/test_feature_pca.npy')

num_instance = test_latent.shape[0]
test_latent = np.load('./latent/wavenet/test_feature.npy')
num_timestep = test_latent.shape[1]

test_latent = test_latent.reshape(num_instance, -1)
test_latent = Isomap(n_components=128).fit_transform(test_latent)

test_label = np.load('./latent/wavenet/test_label.npy')
connected = cal_connected(test_latent)
W, T = cal_segmentation_default(test_latent, connected)

num_round=_num_rounds
chosen_label=np.random.choice([5, 6, 7, 8, 9], 1)[0]
print(chosen_label)
Q, Y, S = run_segmentation(test_label, W, T, num_round, target_label=chosen_label)

TP = 0
FP = 0
FN = 0
TN = 0

S = S.astype(int)
binary_label = (test_label == chosen_label).astype(int)
for i in range(test_label.shape[0]):
    if binary_label[i] == S[i, 0] and binary_label[i] == 1:
        TP += 1
    elif S[i, 0] == 1 and binary_label[i] == 0:
        FP += 1
    elif S[i, 0] == 0 and binary_label[i] == 1:
        FN += 1
    else:
        TN += 1
print('TP: {}, FP: {}, FN: {}, TN: {}, Sum: {}'.format(TP, FP, FN, TN, TP + FP + FN + TN))
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('Precision: {:.2f}, Recall: {:.2f}, Accuracy: {:.2f}'.format(precision, recall, (TP + TN) / (TP + FP + FN + TN)))

#####################################################################################################################

# import argparse
# from meta import Meta
# argparser = argparse.ArgumentParser()
# argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
# argparser.add_argument('--n_way', type=int, help='n way', default=2)
# argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=30)
# argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=30)
# argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
# argparser.add_argument('--imgc', type=int, help='imgc', default=16)
# argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
# argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
# argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
# argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
# argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
# args = argparser.parse_args()
# config = [
#     ('conv2d', [32, 16, 3, 1, (2, 1), 0]),
#     ('relu', [True]),
#     ('bn', [32]),
#     ('max_pool2d', [2, 2, 0]),
#     ('conv2d', [64, 32, 3, 1, (2, 1), 0]),
#     ('relu', [True]),
#     ('bn', [64]),
#     ('max_pool2d', [2, 2, 0]),
#     ('conv2d', [128, 64, 3, 1, (2, 1), 0]),
#     ('relu', [True]),
#     ('bn', [128]),
#     ('max_pool2d', [2, 2, 0]),
#     ('flatten', []),
#     ('linear', [args.n_way, 1408])
# ]
# device = torch.device('cuda')
# maml = Meta(args, config).to(device)
# maml.load_state_dict(torch.load("maml.pth"))

# test_latent = np.load('./latent/wavenet/test_feature_tsne.npy')
# test_feature = np.load('./latent/wavenet/test_feature.npy')
# test_label = np.load('./latent/wavenet/test_label.npy')

# num_instance = test_latent.shape[0]
# test_latent = np.load('./latent/wavenet/test_feature.npy')
# num_timestep = test_latent.shape[1]
# test_latent = test_latent.reshape(num_instance, -1)
# test_latent = Isomap(n_components=256).fit_transform(test_latent)

# connected = cal_connected(test_latent)
# W, T = cal_segmentation_default(test_latent, connected)

# num_round=_num_rounds
# chosen_label=np.random.choice([5, 6, 7, 8, 9], 1)[0]
# Q, Y, S = run_segmentation(test_label, W, T, num_round, target_label=chosen_label)

# support_x = []
# support_y = []
# query_x = []
# query_y = []

# k_shot = 20
# random_idx = np.random.choice(test_label.shape[0], 1000, False)
# positive = 0
# negative = 0
# for i in range(random_idx.shape[0]):
#     idx = random_idx[i]
#     if S[idx, 0] == 1 and positive != k_shot:
#         support_x.append(test_feature[idx])
#         support_y.append(1)
#         positive += 1
#     elif S[idx, 0] == 0 and negative != k_shot:
#         support_x.append(test_feature[idx])
#         support_y.append(0)
#         negative += 1
    
#     if positive == k_shot and negative == k_shot:
#         break
# support_x = np.array(support_x)
# support_y = np.array(support_y)
# support_x = torch.from_numpy(support_x).float().permute(0, 2, 1).unsqueeze(-1).to(device)
# support_y = torch.from_numpy(support_y).long().to(device)

# print(support_x.shape)
# print(support_y.shape)

# random_idx = np.random.choice(test_label.shape[0], 1000, False)
# positive = 0
# negative = 0
# for i in range(random_idx.shape[0]):
#     idx = random_idx[i]
#     if test_label[idx] == chosen_label and positive != k_shot:
#         query_x.append(test_feature[idx])
#         query_y.append(1)
#         positive += 1
#     elif test_label[idx] != chosen_label and negative != k_shot:
#         query_x.append(test_feature[idx])
#         query_y.append(0)
#         negative += 1
    
#     if positive == k_shot and negative == k_shot:
#         break
# query_x = np.array(query_x)
# query_y = np.array(query_y)
# query_x = torch.from_numpy(query_x).float().permute(0, 2, 1).unsqueeze(-1).to(device)
# query_y = torch.from_numpy(query_y).long().to(device)

# print(query_x.shape)
# print(query_y.shape)

# accs, net = maml.finetunningNet(support_x, support_y, query_x, query_y)
# net.eval()
# print(accs)

# test_batch_size = 64
# iterations = int(np.ceil(test_feature.shape[0] / test_batch_size))
# predictions = []
# for i in range(iterations):
#     batch = test_feature[i*test_batch_size:min((i+1)*test_batch_size, test_feature.shape[0])]
#     batch = torch.from_numpy(batch).float().permute(0, 2, 1).unsqueeze(-1).to(device)
#     output = net(batch).argmax(-1)
#     predictions.append(output)
# predictions = torch.cat(predictions, 0)
# predictions = predictions.detach().cpu().numpy()
# # print(predictions)
# # print((test_label == chosen_label).astype(int))

# TP = 0
# FP = 0
# FN = 0
# TN = 0

# binary_label = (test_label == chosen_label).astype(int)
# for i in range(test_label.shape[0]):
#     if binary_label[i] == predictions[i] and binary_label[i] == 1:
#         TP += 1
#     elif predictions[i] == 1 and binary_label[i] == 0:
#         FP += 1
#     elif predictions[i] == 0 and binary_label[i] == 1:
#         FN += 1
#     else:
#         TN += 1
# print('TP: {}, FP: {}, FN: {}, TN: {}, Sum: {}'.format(TP, FP, FN, TN, TP + FP + FN + TN))
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# print('Precision: {:.2f}, Recall: {:.2f}, Accuracy: {:.2f}'.format(precision, recall, (TP + TN) / (TP + FP + FN + TN)))
