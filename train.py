import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import *

num_epochs = 10
batch_size = 64
device = torch.device("cuda:1")

train_spectrogram = np.load('./spectrogram/train/train_spectrogram.npy')
train_label = np.load('./spectrogram/train/train_label.npy')

val_spectrogram = np.load('./spectrogram/val/val_spectrogram.npy')
val_label = np.load('./spectrogram/val/val_label.npy')

num_iterations_train = int(np.ceil(train_spectrogram.shape[0] / batch_size))
num_iterations_test = int(np.ceil(val_spectrogram.shape[0] / batch_size))

net = NetLatent().to(device)
optimizer = optim.Adam(net.parameters(),lr=0.001)

# val_acc = 0.0
# for epoch in tqdm(range(num_epochs)):
#     loss = 0
#     net.train()
#     for round in range(num_iterations_train):
#         X = train_spectrogram[round*batch_size:min((round+1)*batch_size, train_spectrogram.shape[0])]
#         X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
#         y = train_label[round*batch_size:min((round+1)*batch_size, train_spectrogram.shape[0])]

#         optimizer.zero_grad()
#         output = net(torch.from_numpy(X).float().to(device))
#         loss = F.nll_loss(output, torch.from_numpy(y).to(device))
#         loss.backward()
#         optimizer.step()

#     net.eval()
#     total_train = 0
#     correct_train = 0
#     for round in range(num_iterations_train):
#         X = train_spectrogram[round*batch_size:min((round+1)*batch_size, train_spectrogram.shape[0])]
#         X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
#         y = train_label[round*batch_size:min((round+1)*batch_size, train_spectrogram.shape[0])]

#         output = net(torch.from_numpy(X).float().to(device))

#         total_train += X.shape[0]
#         correct_train += (output.argmax(dim=1).detach().cpu().numpy() == y).astype(int).sum()

#     total = 0
#     correct = 0
#     for round in range(num_iterations_test):
#         X = val_spectrogram[round*batch_size:min((round+1)*batch_size, val_spectrogram.shape[0])]
#         X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
#         y = val_label[round*batch_size:min((round+1)*batch_size, val_spectrogram.shape[0])]

#         output = net(torch.from_numpy(X).float().to(device))

#         total += X.shape[0]
#         correct += (output.argmax(dim=1).detach().cpu().numpy() == y).astype(int).sum()

#     print("Epoch: {}, Loss: {}, Accuracy(test): {}%, Accuracy(train): {}%".format(epoch, loss.detach().cpu().numpy(), correct*100/total, correct_train*100/total_train))

#     if correct*100/total > val_acc:
#         print('Saving Model')
#         val_acc = correct*100/total
#         torch.save(net.state_dict(), "classifier.pth")

net.load_state_dict(torch.load('classifier.pth'))
net.eval()

val_latent = []
for round in range(num_iterations_test):
    X = val_spectrogram[round*batch_size:min((round+1)*batch_size, val_spectrogram.shape[0])]
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    y = val_label[round*batch_size:min((round+1)*batch_size, val_spectrogram.shape[0])]

    latent = net.get_latent(torch.from_numpy(X).float().to(device))
    val_latent.append(latent.detach().cpu())
val_latent = torch.cat(val_latent, 0).numpy()

start = time.time()
X_embedded = TSNE(n_components=2).fit_transform(val_latent)
print('Calculating TSNE tooks: {} seconds'.format(time.time()-start))

colors = []
for i in range(val_label.shape[0]):
    if val_label[i] == 0:
        colors.append('blue')
    elif val_label[i] == 1:
        colors.append('red')
    elif val_label[i] == 2:
        colors.append('yellow')
    elif val_label[i] == 3:
        colors.append('green')
    elif val_label[i] == 4:
        colors.append('black')
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
plt.title("Train Feature Tsne Visualization")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
