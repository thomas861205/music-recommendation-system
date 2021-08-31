import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NetLatent(nn.Module):
    def __init__(self):
        super(NetLatent, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, (1, 2), (1, 1))
        self.conv2 = nn.Conv2d(16, 32, 3, (1, 2), (1, 1))
        self.conv3 = nn.Conv2d(32, 64, 3, (1, 2), (1, 1))
        self.conv4 = nn.Conv2d(64, 64, 3, (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(64, 32, 3, (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(32, 16, 3, (1, 1), (1, 1))
        self.fc1 = nn.Linear(16*8*10, 256)
        self.fc2 = nn.Linear(256, 5)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def get_latent(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2(x)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
