import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import *

batch_size = 64
device = torch.device("cuda:1")

net = NetLatent().to(device)
net.load_state_dict(torch.load('classifier.pth'))
net.eval()

####################################################################################################################################################

# train_spectrogram = np.load('./spectrogram/train/train_spectrogram.npy')
# train_label = np.load('./spectrogram/train/train_label.npy')

# num_iterations_train = int(np.ceil(train_spectrogram.shape[0] / batch_size))

# train_latent = []
# for iteration in range(num_iterations_train):
#     X = train_spectrogram[iteration*batch_size:min((iteration+1)*batch_size, train_spectrogram.shape[0])]
#     X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
#     y = train_label[iteration*batch_size:min((iteration+1)*batch_size, train_spectrogram.shape[0])]

#     latent = net.get_latent(torch.from_numpy(X).float().to(device))
#     train_latent.append(latent.detach().cpu())
# train_latent = torch.cat(train_latent, 0).numpy()
# np.save('./latent/classifier/train_latent.npy', train_latent)
# np.save('./latent/classifier/train_label.npy', train_label)
# exit()

####################################################################################################################################################

# test_spectrogram = np.load('./spectrogram/test/test_spectrogram.npy')
# test_label = np.load('./spectrogram/test/test_label.npy')

# num_iterations_test = int(np.ceil(test_spectrogram.shape[0] / batch_size))

# test_latent = []
# for iteration in range(num_iterations_test):
#     X = test_spectrogram[iteration*batch_size:min((iteration+1)*batch_size, test_spectrogram.shape[0])]
#     X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
#     y = test_label[iteration*batch_size:min((iteration+1)*batch_size, test_spectrogram.shape[0])]

#     latent = net.get_latent(torch.from_numpy(X).float().to(device))
#     test_latent.append(latent.detach().cpu())
# test_latent = torch.cat(test_latent, 0).numpy()
# np.save('./latent/classifier/test_latent.npy', test_latent)
# np.save('./latent/classifier/test_label.npy', test_label)

# start = time.time()
# X_embedded = TSNE(n_components=2).fit_transform(test_latent)
# print('Calculating TSNE tooks: {} seconds'.format(time.time()-start))

# colors = []
# for i in range(test_label.shape[0]):
#     if test_label[i] == 5:
#         colors.append('blue')
#     elif test_label[i] == 6:
#         colors.append('red')
#     elif test_label[i] == 7:
#         colors.append('yellow')
#     elif test_label[i] == 8:
#         colors.append('green')
#     elif test_label[i] == 9:
#         colors.append('black')
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
# plt.title("Test Feature Tsne Visualization")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

####################################################################################################################################################

test_feature = np.load('./latent/wavenet/test_feature.npy')
test_label = np.load('./latent/wavenet/test_label.npy')

new_test_feature = []
for i in tqdm(range(test_feature.shape[-1])):
    temp = test_feature[:, :, i]
    temp = PCA(n_components=128).fit_transform(temp)
    # temp = TSNE(n_components=3).fit_transform(temp)
    new_test_feature.append(temp)
new_test_feature = np.array(new_test_feature).transpose(1, 2, 0)
test_feature = new_test_feature.reshape(test_feature.shape[0], -1)
# np.save('./latent/wavenet/test_feature_tsne.npy', test_feature)
np.save('./latent/wavenet/test_feature_pca.npy', test_feature)

# num_instance = test_feature.shape[0]
# test_latent = np.load('./latent/wavenet/test_feature.npy')
# num_timestep = test_latent.shape[1]
# test_latent = test_latent.reshape(num_instance, -1)
# test_latent = Isomap(n_components=256).fit_transform(test_latent)
# test_feature = test_latent

test_feature = test_feature.reshape(test_feature.shape[0], -1)
start = time.time()
X_embedded = TSNE(n_components=2).fit_transform(test_feature)
# X_embedded = PCA(n_components=2).fit_transform(test_feature)
print('Calculating TSNE tooks: {} seconds'.format(time.time()-start))

colors = []
for i in range(test_label.shape[0]):
    if test_label[i] == 5:
        colors.append('blue')
    elif test_label[i] == 6:
        colors.append('red')
    elif test_label[i] == 7:
        colors.append('yellow')
    elif test_label[i] == 8:
        colors.append('green')
    elif test_label[i] == 9:
        colors.append('black')
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
plt.title("Test Feature Tsne Visualization")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

####################################################################################################################################################

test_feature = np.load('./latent/wavenet/train_feature.npy')
test_label = np.load('./latent/wavenet/train_label.npy')

new_test_feature = []
for i in tqdm(range(test_feature.shape[-1])):
    temp = test_feature[:, :, i]
    temp = PCA(n_components=128).fit_transform(temp)
    # temp = TSNE(n_components=3).fit_transform(temp)
    new_test_feature.append(temp)
new_test_feature = np.array(new_test_feature).transpose(1, 2, 0)
test_feature = new_test_feature.reshape(test_feature.shape[0], -1)
# np.save('./latent/wavenet/train_feature_tsne.npy', test_feature)
np.save('./latent/wavenet/train_feature_pca.npy', test_feature)

# num_instance = test_feature.shape[0]
# test_latent = np.load('./latent/wavenet/test_feature.npy')
# num_timestep = test_latent.shape[1]
# test_latent = test_latent.reshape(num_instance, -1)
# test_latent = Isomap(n_components=256).fit_transform(test_latent)
# test_feature = test_latent

test_feature = test_feature.reshape(test_feature.shape[0], -1)
start = time.time()
X_embedded = TSNE(n_components=2).fit_transform(test_feature)
# X_embedded = PCA(n_components=2).fit_transform(test_feature)
print('Calculating TSNE tooks: {} seconds'.format(time.time()-start))

colors = []
for i in range(test_label.shape[0]):
    if test_label[i] == 0:
        colors.append('blue')
    elif test_label[i] == 1:
        colors.append('red')
    elif test_label[i] == 2:
        colors.append('yellow')
    elif test_label[i] == 3:
        colors.append('green')
    elif test_label[i] == 4:
        colors.append('black')
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
plt.title("Train Feature Tsne Visualization")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
