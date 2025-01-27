# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import gzip
import torch
import numpy as np


data_dir = '../data'

def load_data(dataset_type ='train', mode='binary'):
   
    labels_path = os.path.join(data_dir, f'{dataset_type}-labels-idx1-ubyte')
    images_path = os.path.join(data_dir, f'{dataset_type}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)  # Skip the header
        y_data = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)  # Skip the header
        x_data = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(y_data), 784)

    if mode == 'binary':
        y_data = torch.tensor([1 if label % 2 == 0 else -1 for label in y_data])
    else:
        y_data = torch.tensor(y_data)
    
    return x_data, y_data

def evaluate(weights, mode='binary'):
    
    x_test, y_test = load_data("t10k", mode=mode)  # Load test data
    correct = 0
    total = len(x_test)
    
    for i in range(total):
        prediction = np.sign(np.dot(x_test[i], weights))  # Get prediction
        if prediction == np.sign(y_test[i]):  # Binary comparison
            correct += 1

    return correct / total  # Return accuracy as a percentage



def run_algorithm(iteration, update_fn, dataset_type="train", use_averaging=False , mode = 'binary'):
    """
    Final weights, mistake list, train accuracy list, test accuracy list
    """
    x_train, y_train = load_data(dataset_type)
    weights = np.zeros(x_train.shape[1])
    
    mistake_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    if use_averaging:
        avg_weights = np.zeros(x_train.shape[1])
        count = 1  # Counter to compute average weights

    for i in range(iteration):
        mistake, correct = 0, 0
        for j in range(len(x_train)):
            if np.sign(np.dot(x_train[j], weights)) != np.sign(y_train[j]):
                mistake += 1
                
                if use_averaging:
                    # For Averaged Perceptron, pass avg_weights and count to update_fn
                    weights, avg_weights, count = update_fn(weights, x_train[j], y_train[j], avg_weights, count)
                else:
                    # For Standard Perceptron, no need to pass avg_weights and count
                    weights = update_fn(weights, x_train[j], y_train[j])
            else:
                correct += 1

        mistake_list.append(mistake)
        train_accuracy_list.append(correct / len(x_train))

        if use_averaging:
            final_weights = avg_weights / count
        else:
            final_weights = weights

        test_accuracy_list.append(evaluate(final_weights,mode))

    return final_weights, mistake_list, train_accuracy_list, test_accuracy_list


# def increment_run(iteration, update_fn, size):
#     """
#     Run the algorithm incrementally, saving performance every `size` samples.
    
#     :param iteration: Number of iterations
#     :param update_fn: Function to update the weights
#     :param size: Size of incremental steps
#     :return: (weights, data_size, test_accuracy_list)
#     """
#     x_train, y_train = load_data("train")
#     weights = np.zeros(x_train.shape[1])
    
#     data_size = []
#     test_accuracy_list = []
    
#     for i in range(iteration):
#         for j in range(len(x_train)):
#             if np.sign(np.dot(x_train[j], weights)) != np.sign(y_train[j]):
#                 weights = update_fn(weights, x_train[j], y_train[j])
            
#             if (j + 1) % size == 0:
#                 test_accuracy_list.append(evaluate(weights))
#                 data_size.append((i+1) * (j+1))
    
#     return weights, data_size, test_accuracy_list

def increment_run(iteration, update_fn, size, use_averaging=False):
    
    x_train, y_train = load_data("train")
    weights = np.zeros(x_train.shape[1])

    if use_averaging:
        avg_weights = np.zeros(x_train.shape[1])
        count = 1  # Counter for averaging weights

    data_size = []
    test_accuracy_list = []
    
    for i in range(iteration):
        for j in range(len(x_train)):
            if np.sign(np.dot(x_train[j], weights)) != np.sign(y_train[j]):
                if use_averaging:
                    weights, avg_weights, count = update_fn(weights, x_train[j], y_train[j], avg_weights, count)
                else:
                    weights = update_fn(weights, x_train[j], y_train[j])

            if (j + 1) % size == 0:
                if use_averaging:
                    final_weights = avg_weights / count
                else:
                    final_weights = weights
                
                test_accuracy_list.append(evaluate(final_weights))
                data_size.append((i + 1) * (j + 1))

    if use_averaging:
        return avg_weights / count, data_size, test_accuracy_list
    else:
        return weights, data_size, test_accuracy_list

# Perceptron Algorithm
def perceptron_update(weights, xt, yt):
    return np.add(weights, yt * xt)

def run_perceptron(iteration, kind="train"):
    return run_algorithm(iteration, perceptron_update, kind)


# Passive-Aggressive Algorithm
def passive_aggressive_update(weights, xt, yt):
    learning_rate = (1 - yt * np.dot(weights, xt)) / (np.linalg.norm(xt) ** 2)
    return np.add(weights, learning_rate * yt * xt)

def run_passive_aggressive(iteration, kind="train"):
    return run_algorithm(iteration, passive_aggressive_update, kind)



def averaged_perceptron_update(weights, xt, yt, avg_weights, count):
    """
    Update weights for the Averaged Perceptron and compute average weights.
    """
    weights = np.add(weights, yt * xt)
    avg_weights = np.add(avg_weights, weights)
    count += 1
    return weights, avg_weights, count

def averaged_perceptron_increment_run(iteration, size):
    """
    Run the Averaged Perceptron algorithm incrementally, saving performance every `size` samples.
    """
    return increment_run(iteration, averaged_perceptron_update, size, use_averaging=True)


def perceptron_increment_run(iteration, size):
    return increment_run(iteration, perceptron_update, size, use_averaging=False)


def passive_aggressive_increment_run(iteration, size):
    return increment_run(iteration, passive_aggressive_update, size, use_averaging=False)
