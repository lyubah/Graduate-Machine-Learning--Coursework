import numpy as np
import os

# Define the path to your data
data_dir = '../data'


# Load data function for multi-class classification
def load_data(dataset_type='train', num_classes=10):
    
    labels_path = os.path.join(data_dir, f'{dataset_type}-labels-idx1-ubyte')
    images_path = os.path.join(data_dir, f'{dataset_type}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        y_data = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        x_data = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(y_data), 784)

    return x_data, y_data



# Evaluation function for multi-class classification
def evaluate(weights, num_classes=10):
    
    x_test, y_test = load_data("t10k", num_classes)
    correct = 0
    total = len(x_test)
    
    for i in range(total):
        all_scores = [np.dot(weights, get_weight_vector(x_test[i], k, num_classes)) for k in range(num_classes)]
        y_pred = np.argmax(all_scores)
        if y_pred == y_test[i]:
            correct += 1

    return correct / total




# Weight-vector representation for multi-class classification
def get_weight_vector(x, y, num_classes=10):
    vector = np.zeros(len(x) * num_classes)
    for i in range(len(x)):
        vector[len(x) * y + i] = x[i]
    return vector


def run_multi_algorithm(iteration, update_fn, dataset_type="train", num_classes=10):
   
    # Load the dataset
    x_train, y_train = load_data(dataset_type, num_classes)

    # Initialize weights for multi-class classification
    weights = np.zeros(len(x_train[0]) * num_classes)
    
    # Lists to track mistakes and accuracy over iterations
    mistake_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    for i in range(iteration):
        mistake, correct = 0, 0
        
        # Iterate through each sample
        for j in range(len(x_train)):
            # Compute scores for each class
            all_scores = [np.dot(weights, get_weight_vector(x_train[j], k, num_classes)) for k in range(num_classes)]
            y_pred = np.argmax(all_scores)  # Get the predicted label

            # If the prediction is incorrect
            if y_pred != y_train[j]:
                mistake += 1
                weights = update_fn(weights, x_train[j], y_train[j], y_pred, num_classes)
            else:
                correct += 1

        # Track mistakes and accuracies after each iteration
        mistake_list.append(mistake)
        train_accuracy_list.append(correct / len(x_train))
        test_accuracy_list.append(evaluate(weights, num_classes))

    return weights, mistake_list, train_accuracy_list, test_accuracy_list


# Update weights for Perceptron
def update_multi_weights_perceptron(weight_vector, xt, yt, y_pred, num_classes=10):
    return weight_vector + get_weight_vector(xt, yt, num_classes) - get_weight_vector(xt, y_pred, num_classes)


def update_multi_weights_pa(weight_vector, xt, yt, y_pred, num_classes=10):
    denom = np.linalg.norm(get_weight_vector(xt, yt, num_classes) - get_weight_vector(xt, y_pred, num_classes)) + 1e-8
    learning_rate = (1 - (np.dot(weight_vector, get_weight_vector(xt, yt, num_classes)) - np.dot(weight_vector, get_weight_vector(xt, y_pred, num_classes)))) / denom

    # Debugging output
    if np.isnan(learning_rate) or np.isinf(learning_rate):
        print(f"Warning: Invalid learning rate: {learning_rate}")
        print(f"Weight vector: {weight_vector}, xt: {xt}, yt: {yt}, y_pred: {y_pred}")

    return weight_vector + learning_rate * (get_weight_vector(xt, yt, num_classes) - get_weight_vector(xt, y_pred, num_classes))




# Multi-class Perceptron
def run_multi_perceptron(iteration, dataset_type="train", num_classes=10):
   
    return run_multi_algorithm(iteration, update_multi_weights_perceptron, dataset_type, num_classes)


# Multi-class Passive-Aggressive Algorithm
# def run_multi_passive_aggressive(iteration, dataset_type="train", num_classes=10):
#     """
#     Run the multi-class Passive-Aggressive algorithm.
    
#     :param iteration: Number of iterations.
#     :param dataset_type: Type of the dataset ('train' or 'test').
#     :param num_classes: Number of classes for multi-class classification.
#     :return: Final weights, mistake list, train accuracy list, test accuracy list.
#     """
#     return run_multi_algorithm(iteration, update_multi_weights_pa, dataset_type, num_classes)


# Incremental run for Perceptron and Passive-Aggressive algorithms
def multi_increment_run(iteration, size, update_fn, num_classes=10):
    # Load the dataset
    x_train, y_train = load_data("train", num_classes)
    
    # Initialize weights for multi-class classification
    weight_vector = np.zeros(len(x_train[0]) * num_classes)
    
    # Lists to track data size and test accuracy
    data_size = []
    test_accuracy_list = []
    
    for i in range(iteration):
        print(f"Iteration: {i}")
        for j in range(len(x_train)):
            # Compute scores for each class
            all_scores = [np.dot(weight_vector, get_weight_vector(x_train[j], k, num_classes)) for k in range(num_classes)]
            y_pred = np.argmax(all_scores)  # Get the predicted label
            
            # If the prediction is incorrect, update weights
            if y_pred != y_train[j]:
                weight_vector = update_fn(weight_vector, x_train[j], y_train[j], y_pred, num_classes)
            
            # Every 'size' number of samples, track the test accuracy
            if (j + 1) % size == 0:
                test_accuracy_list.append(evaluate(weight_vector, num_classes))
                data_size.append((i + 1) * (j + 1))
    
    return weight_vector, data_size, test_accuracy_list

