import os
import math
import numpy as np
import random
from PIL import Image
import torch
from tqdm import tqdm
from numpy.linalg import inv
from sklearn.preprocessing import normalize
from scipy.spatial import distance

def correlation_distance(vec1, vec2):
    avg_1 = np.sum(vec1) / vec1.shape[0]
    avg_2 = np.sum(vec2) / vec2.shape[0]
    dividend = np.dot(np.transpose(vec1 - avg_1), (vec2 - avg_2))
    divisor = np.sqrt(np.dot(np.transpose(vec1 - avg_1), (vec1 - avg_1))) * np.sqrt(np.dot(np.transpose(vec2 - avg_2), (vec2 - avg_2)))
    return 1 - dividend / divisor

def cal_connected(latents):
    distance_matrix = distance.cdist(latents, latents, 'euclidean')

    connected = []
    for i in range(distance_matrix.shape[0]):
        idx = np.argsort(distance_matrix[i])
        connected.append(idx[1:30])
    return connected

def cal_segmentation_default(X_input, connected):
    total_size = X_input.shape[0]

    print("Calculating W matrix")
    W = distance.cdist(X_input, X_input, 'minkowski', p=1.)

    W = W / (2 * 0.0913 * 0.0913)
    W /= W.max()
    W = np.exp(-1 * W)

    for i in range(total_size):
        disconnected = list(set([j for j in range(total_size)]) - set(connected[i]))
        W[i, disconnected] = 0
    W = np.abs(W)

    print("Calculating D matrix")
    D = np.zeros((total_size, total_size), np.float32)
    for i in range(total_size):
        D[i, i] = np.sum(W[i])

    print("Calculating T matrix")
    alpha = 0.99
    D_inverse_half = np.zeros((total_size, total_size), dtype=np.float32)
    for i in range(total_size):
        D_inverse_half[i, i] = 1 / np.sqrt(D[i, i])
    T = np.eye(total_size, dtype=np.float32) - alpha * np.matmul(np.matmul(D_inverse_half, W), D_inverse_half)
    T = inv(T)
    T = normalize(T, norm='l1', axis=0, copy=True, return_norm=False)

    return W, T

def init_segmentation_label(gt_label, chosen_label, num_songs=5):
    total_size = gt_label.shape[0]
    qualified = []
    label = np.zeros(total_size)
    label = label.astype(int)
    for i in range(total_size):
        if gt_label[i] == chosen_label:
            qualified.append(i)
            label[i] = 1

    qualified = np.array(qualified)
    label = np.array(label)
    index = np.random.choice(qualified.shape[0], num_songs, replace=False)
    selected_number = qualified[index]
    Q = selected_number.tolist()

    Y = np.zeros((total_size, 2))
    for num in selected_number:
        Y[num][0] = 1
        Y[num][1] = 0

    return Q, Y

def run_segmentation(Y_input, W, T, rounds, target_label=1):
    Q = []
    Y = np.zeros((Y_input.shape[0], 2))

    Y_label = Y_input.reshape(-1, 1)
    total_size = Y_label.shape[0]

    S = np.zeros((total_size, 2), dtype=np.float32)

    for i in range(rounds):
        if i == 0:
            choice = random.randint(0, total_size - 1)
            Q.append(choice)
            prediction = 1

            if (Y_label[choice, 0] == target_label):
                prediction = 0

            if prediction == 0:
                Y[choice, 0] = 1
                Y[choice, 1] = 0
            else:
                Y[choice, 0] = 0
                Y[choice, 1] = 1
        else:
            choice, S = inference_query(W, T, Q, Y)
            prediction = 1

            if (Y_label[choice, 0] == target_label):
                prediction = 0

            if prediction == 0:
                Y[choice, 0] = 1
                Y[choice, 1] = 0
            else:
                Y[choice, 0] = 0
                Y[choice, 1] = 1
    return Q, Y, S

def inference_query(W, T, Q, Y):
    total_size = W.shape[0]
    D = np.zeros((total_size, total_size), np.float32)
    for i in range(total_size):
        D[i, i] = np.sum(W[i])
    D_inverse_half = np.zeros((total_size, total_size), dtype=np.float32)
    for i in range(total_size):
        D_inverse_half[i, i] = 1 / np.sqrt(D[i, i])

    #Initialize S matrix and assume at first all points belong to ~R, initialize homogeneous case to true
    S = np.zeros((total_size, 2), dtype=np.float32)
    S[:, 1] = 1
    homo = True

    #Test to see if homogeneous feeback or heterogeneous feedback
    if np.sum(Y[:, 0]) < 1.0 or np.sum(Y[:, 1]) < 1.0:
        homo = True
    else:
        homo = False

    #Change S matrix depending on homogeneous feed back or heterogeneous feedback
    if homo:
        for i in range(total_size):
            idx = np.argmax(T[:, i])
            if (T[idx, i] > 1/total_size):
                S[i, 0] = 1
                S[i, 1] = 0
    else:
        beta = 0.9
        T_pred = np.eye(total_size, dtype=np.float32) - beta * np.matmul(np.matmul(D_inverse_half, W), D_inverse_half)
        T_pred = np.matmul(inv(T_pred), Y)
        T_pred = normalize(T_pred, norm='l1', axis=1, copy=True, return_norm=False)
        T_pred = np.abs(T_pred)

        for i in range(total_size):
            if T_pred[i, 0] > T_pred[i, 1]:
                S[i, 0] = 1
                S[i, 1] = 0

    #Choose what to query next
    not_Q = list(set([j for j in range(total_size)]) - set(Q))
    if homo:
        idx = -1
        max_val = -1
        for i in not_Q:
            temp = 0
            for j in Q:
                correlation = correlation_distance(T[:, i], T[:, j])
                temp += correlation
            if temp > max_val:
                max_val = temp
                idx = i
        Q.append(idx)
        return idx, S
    else:
        idx = -1
        min_val = 2147483640
        for i in not_Q:
            temp = max(T_pred[i, 0], T_pred[i, 1])
            if temp < min_val:
                min_val = temp
                idx = i
        Q.append(idx)
        return idx, S
