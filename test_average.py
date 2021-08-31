import os
import pickle
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import *

device = torch.device("cpu")
net = NetLatent()
net.load_state_dict(torch.load("classifier.pth", map_location='cpu'))
net.to(device)
net.eval()

# song_names, latents = load_latents(net, device)
# latents = latents.numpy()
# with open('song_name.pkl', 'wb') as f:
#     pickle.dump(song_names, f)
# np.save("latents.npy", latents)

with open('song_name.pkl', 'rb') as f:
    song_names = pickle.load(f)
latents = np.load("latents.npy")

user_latents = []
files = [data for data in os.listdir("./user") if data.endswith(".npy")]
for i in range(len(files)):
    user_latents.append(latents[song_names.index(files[i])])
user_latents = np.array(user_latents)

average_latent = np.zeros(user_latents.shape[1])
for i in range(user_latents.shape[0]):
    average_latent += user_latents[i]
average_latent /= user_latents.shape[0]
#average_latent = average_latent.reshape(1, -1)

recommend = []
for i in range(latents.shape[0]):
    # temp = cosine_similarity(average_latent, latents[i].reshape(1, -1))
    temp = np.sqrt(np.sum((average_latent - latents[i]) ** 2))
    if temp < 70:
        recommend.append(song_names[i])
        print(song_names[i])

total = len(recommend)
correct = 0
for i in range(len(recommend)):
    if song_names[i].split("_")[0] == "bigroom":
        correct += 1
print(correct)
print("Accuracy: {}%".format(100 * correct / total))
