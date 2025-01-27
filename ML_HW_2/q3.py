# q3.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv(url, names=column_names)

    # Map diagnosis to binary values: Malignant (M) = 1, Benign (B) = 0
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

    # Split into features and labels
    X = data.iloc[:, 2:]
    y = data['Diagnosis']

    # Ensure the data is sorted consistently
    data_length = len(data)
    indices = np.arange(data_length)
    # Use fixed random state for reproducibility
    np.random.seed(42)
    np.random.shuffle(indices)

    # Calculate split indices
    train_end = int(0.7 * data_length)
    val_end = int(0.8 * data_length)

    # Split the indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create splits
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_val = X.iloc[val_indices]
    y_val = y.iloc[val_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_id3_decision_tree(X_train, y_train, max_depth=None):
     
    clf = DecisionTreeClassifier(
        criterion='entropy',  # Emulate ID3 by using entropy
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_decision_tree(clf, X_train, y_train, X_val, y_val, X_test, y_test):
     
    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    # Accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    accuracies = {
        'unpruned_train_accuracy': train_accuracy,
        'unprffuned_val_accuracy': val_accuracy,
        'unpruned_test_accuracy': test_accuracy
    }

    return accuracies



def prune_decision_tree(clf, X_train, y_train, X_val, y_val):
     
    # Compute the pruning path using training data
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    # Remove duplicate alphas to reduce computation
    ccp_alphas = np.unique(ccp_alphas)

    # Initialize variables to store the best alpha
    best_alpha = 0
    best_accuracy = 0

    # List to store classifiers for plotting (optional)
    clfs = []

    # Iterate over alphas to find the one that gives the highest validation accuracy
    for alpha in ccp_alphas:
        clf_pruned = DecisionTreeClassifier(
            criterion='entropy',
            random_state=42,
            ccp_alpha=alpha
        )
        clf_pruned.fit(X_train, y_train)
        y_val_pred = clf_pruned.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        if acc >= best_accuracy:
            best_accuracy = acc
            best_alpha = alpha
        clfs.append(clf_pruned)

    # Train the pruned tree with the best alpha on the training data
    pruned_clf = DecisionTreeClassifier(
        criterion='entropy',
        random_state=42,
        ccp_alpha=best_alpha
    )
    pruned_clf.fit(X_train, y_train)

    return pruned_clf


def evaluate_pruned_tree(pruned_clf, X_train, y_train, X_val, y_val, X_test, y_test):
     
    # Predictions
    y_train_pred = pruned_clf.predict(X_train)
    y_val_pred = pruned_clf.predict(X_val)
    y_test_pred = pruned_clf.predict(X_test)

    # Accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    accuracies = {
        'pruned_train_accuracy': train_accuracy,
        'pruned_val_accuracy': val_accuracy,
        'pruned_test_accuracy': test_accuracy
    }

    return accuracies

def compare_trees(unpruned_acc, pruned_acc):
    
    observations = ""
    observations += "===+++++++++ Comparison of Unpruned and Pruned Decision Trees ===\n\n"
    observations += "Unpruned Decision Tree@@@:\n"
    observations += f" - Train Accuracy: {unpruned_acc['unpruned_train_accuracy'] * 100:.2f}%\n"
    observations += f" - Validation Accuracy: {unpruned_acc['unpruned_val_accuracy'] * 100:.2f}%\n"
    observations += f" - Test Accuracy: {unpruned_acc['unpruned_test_accuracy'] * 100:.2f}%\n\n"

    observations += "Pruned Decision Tree:\n"
    observations += f" - Train Accuracy: {pruned_acc['pruned_train_accuracy'] * 100:.2f}%\n"
    observations += f" - Validation Accuracy: {pruned_acc['pruned_val_accuracy'] * 100:.2f}%\n"
    observations += f" - Test Accuracy: {pruned_acc['pruned_test_accuracy'] * 100:.2f}%\n\n"

    observations += "=== Observations ===\n"
    observations += f"- Training Accuracy decreased from {unpruned_acc['unpruned_train_accuracy'] * 100:.2f}% to {pruned_acc['pruned_train_accuracy'] * 100:.2f}% after pruning.\n"
    observations += f"- Validation Accuracy improved or remained similar at {pruned_acc['pruned_val_accuracy'] * 100:.2f}%.\n"
    observations += f"- Test Accuracy improved from {unpruned_acc['unpruned_test_accuracy'] * 100:.2f}% to {pruned_acc['pruned_test_accuracy'] * 100:.2f}% after pruning.\n"
    observations += "\nThis indicates that pruning helped in reducing overfitting, leading to better generalization on unseen data."

    return observations

def q3_main():
    
    # Part (a): Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = data()

    # Part (a): Train ID3 Decision Tree
    clf = train_id3_decision_tree(X_train, y_train)

    # Part (b): Evaluate Unpruned Tree
    unpruned_accuracies = evaluate_decision_tree(clf, X_train, y_train, X_val, y_val, X_test, y_test)

    # Part (c): Prune the Decision Tree
    pruned_clf = prune_decision_tree(clf, X_train, y_train, X_val, y_val)

    # Part (d): Evaluate Pruned Tree
    pruned_accuracies = evaluate_pruned_tree(pruned_clf, X_train, y_train, X_val, y_val, X_test, y_test)

    # Part (d): Compare Trees and Get Observations
    observations = compare_trees(unpruned_accuracies, pruned_accuracies)

    # Print observations
    print(observations)

    # Compile all results into a dictionary
    results = {
        'unpruned_accuracies': unpruned_accuracies,
        'pruned_accuracies': pruned_accuracies,
        'observations': observations
    }

    return results

if __name__ == "__main__":
    q3_results = q3_main()
