import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the path to your data
data_dir = '../data'

# Load and normalize data function for multi-class classification (PA Version)
def load_data_multi_pa(dataset_type='train', num_classes=10):
    labels_path = os.path.join(data_dir, f'{dataset_type}-labels-idx1-ubyte')
    images_path = os.path.join(data_dir, f'{dataset_type}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        y_data = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        x_data = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(y_data), 784)

    # Normalize the data to avoid numerical instability
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)

    return x_data, y_data

# Evaluation function for multi-class classification (PA Version)
def evaluate_multi_pa(weights, num_classes=10):
    x_test, y_test = load_data_multi_pa("t10k", num_classes)
    correct = 0
    total = len(x_test)

    for i in range(total):
        all_scores = [np.dot(weights, get_weight_vector_multi_pa(x_test[i], k, num_classes)) for k in range(num_classes)]
        y_pred = np.argmax(all_scores)
        if y_pred == y_test[i]:
            correct += 1

    return correct / total

# Weight-vector representation for multi-class classification (PA Version, Representation II)
def get_weight_vector_multi_pa(x, y, num_classes=10):
    vector = np.zeros(len(x) * num_classes)
    vector[len(x) * y: len(x) * (y + 1)] = x
    return vector

# Run multi-class algorithm for PA version
def run_multi_pa_algorithm(iteration, dataset_type="train", num_classes=10):
    x_train, y_train = load_data_multi_pa(dataset_type, num_classes)
    weights = np.random.normal(0, 0.01, len(x_train[0]) * num_classes)

    mistake_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    for i in range(iteration):
        mistake, correct = 0, 0

        for j in range(len(x_train)):
            all_scores = [np.dot(weights, get_weight_vector_multi_pa(x_train[j], k, num_classes)) for k in range(num_classes)]
            y_pred = np.argmax(all_scores)

            if y_pred != y_train[j]:
                mistake += 1
                weights = update_multi_weights_pa(weights, x_train[j], y_train[j], y_pred, num_classes)
            else:
                correct += 1

        mistake_list.append(mistake)
        train_accuracy_list.append(correct / len(x_train))
        test_accuracy_list.append(evaluate_multi_pa(weights, num_classes))

    return weights, mistake_list, train_accuracy_list, test_accuracy_list

# Update weights for Passive-Aggressive Algorithm (PA Version)
def update_multi_weights_pa(weight_vector, xt, yt, y_pred, num_classes=10, epsilon=1e-8, max_learning_rate=10.0):
    
    # Calculate weight difference vectors
    diff_vector_yt = get_weight_vector_multi_pa(xt, yt, num_classes)
    diff_vector_y_pred = get_weight_vector_multi_pa(xt, y_pred, num_classes)
    
    # Compute the denominator, adding epsilon for numerical stability
    denom = np.linalg.norm(diff_vector_yt - diff_vector_y_pred) + epsilon
    
    # Compute the margin
    margin = 1 - (np.dot(weight_vector, diff_vector_yt) - np.dot(weight_vector, diff_vector_y_pred))
    
    # Compute learning rate, ensure it's non-negative, and clip it to avoid very large values
    learning_rate = max(0, margin / denom)
    learning_rate = min(learning_rate, max_learning_rate)  # Cap the learning rate
    
    # Debugging output for potential errors
    if np.isnan(learning_rate) or np.isinf(learning_rate):
        print(f"Warning: Invalid learning rate at sample {xt}, label {yt}")
        print(f"Weight vector: {weight_vector}, y_pred: {y_pred}, learning_rate: {learning_rate}")

    # Update the weight vector with the safe learning rate
    return weight_vector + learning_rate * (diff_vector_yt - diff_vector_y_pred)

# Run multi-class Passive-Aggressive Algorithm
def run_multi_passive_aggressive(iteration, dataset_type="train", num_classes=10):
    return run_multi_pa_algorithm(iteration, dataset_type, num_classes)

# Incremental run for Passive-Aggressive algorithm (PA Version)
def multi_increment_run_pa(iteration, size, update_fn, num_classes=10):
    x_train, y_train = load_data_multi_pa("train", num_classes)
    weight_vector = np.random.normal(0, 0.01, len(x_train[0]) * num_classes)

    data_size = []
    test_accuracy_list = []

    for i in range(iteration):
        for j in range(len(x_train)):
            all_scores = [np.dot(weight_vector, get_weight_vector_multi_pa(x_train[j], k, num_classes)) for k in range(num_classes)]
            y_pred = np.argmax(all_scores)

            if y_pred != y_train[j]:
                weight_vector = update_fn(weight_vector, x_train[j], y_train[j], y_pred, num_classes)

            if (j + 1) % size == 0:
                test_accuracy_list.append(evaluate_multi_pa(weight_vector, num_classes))
                data_size.append((i + 1) * (j + 1))

    return weight_vector, data_size, test_accuracy_list
