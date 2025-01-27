# q1c.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def train_svm_poly_kernel(X_train, y_train, C, X_val, y_val, X_test, y_test, degree):
    print(f"\nTraining SVM with polynomial kernel of degree={degree}, C={C}...")
    
    clf = svm.SVC(kernel='poly', degree=degree, C=C, random_state=12, max_iter=10000, tol=1e-3, verbose=False)
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    # Compute accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Number of support vectors
    total_sv = clf.n_support_.sum()
    
    print(f"Degree={degree}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}, Support Vectors={total_sv}")
    
    return train_acc, val_acc, test_acc, total_sv


def plot_accuracies_poly(degrees, accuracies):
    """
    Plots training, validation, and testing accuracies as a function of polynomial degree.

    Parameters:
    - degrees (list): List of polynomial degrees.
    - accuracies (list of tuples): List containing tuples of (train_acc, val_acc, test_acc) for each degree.
    """
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    train_acc = [acc[0] for acc in accuracies]
    val_acc = [acc[1] for acc in accuracies]
    test_acc = [acc[2] for acc in accuracies]
    
    plt.plot(degrees, train_acc, marker='o', label='Training Accuracy', color='tab:blue')
    plt.plot(degrees, val_acc, marker='s', label='Validation Accuracy', color='tab:green')
    plt.plot(degrees, test_acc, marker='^', label='Testing Accuracy', color='tab:red')
    
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Polynomial Degrees')
    plt.xticks(degrees)
    plt.legend()
    
    plt.savefig(os.path.join(os.getcwd(), 'SVM_Accuracy_Polynomial_Degrees.png'))
    plt.show()


def plot_support_vectors_poly(degrees, support_vectors):
    """
    Plots the number of support vectors as a function of polynomial degree.

    Parameters:
    - degrees (list): List of polynomial degrees.
    - support_vectors (list): Number of support vectors for each degree.
    """
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    plt.plot(degrees, support_vectors, marker='D', color='tab:purple', label='Support Vectors')
    
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Number of Support Vectors')
    plt.title('Support Vectors Across Polynomial Degrees')
    plt.xticks(degrees)
    plt.legend()
    
    plt.savefig(os.path.join(os.getcwd(), 'SVM_Support_Vectors_Polynomial_Degrees.png'))
    plt.show()


def q1c_main(C_best, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Executes Part 1(c) of the assignment.
    Trains SVMs with polynomial kernels of degrees 2, 3, and 4 using C_best.
    Returns the best degree based on validation accuracy.

    Parameters:
    - C_best (float): Best C value from Part 1(a).
    - X_train, y_train (np.ndarray): Training data and labels.
    - X_val, y_val (np.ndarray): Validation data and labels.
    - X_test, y_test (np.ndarray): Testing data and labels.

    Returns:
    - best_degree (int): Polynomial degree with the highest validation accuracy.
    """
    degrees = [2, 3, 4]
    accuracies = []
    support_vectors = []
    
    for degree in degrees:
        acc = train_svm_poly_kernel(
            X_train, y_train, C_best, X_val, y_val, X_test, y_test, degree
        )
        accuracies.append(acc[:3])  # (train_acc, val_acc, test_acc)
        support_vectors.append(acc[3])
    
    # Plot accuracies across polynomial degrees
    plot_accuracies_poly(degrees, accuracies)
    
    # Plot number of support vectors across polynomial degrees
    plot_support_vectors_poly(degrees, support_vectors)
    
    # Determine the best degree based on validation accuracy
    val_accuracies = [acc[1] for acc in accuracies]
    max_val_acc = max(val_accuracies)
    best_degree_indices = [i for i, acc in enumerate(val_accuracies) if acc == max_val_acc]
    best_degree_candidates = [degrees[i] for i in best_degree_indices]
    best_degree = min(best_degree_candidates)  # Choose the smallest degree among candidates
    
    print(f"\nBest Polynomial Degree based on Validation Accuracy: {best_degree}")
    
    return best_degree


if __name__ == "__main__":
    """
    Executes Part 1(c) independently.
    Loads and preprocesses data, identifies the best C from Part 1(a),
    trains SVMs with polynomial kernels, and determines the best polynomial degree.
    """
    import q1  # Assuming q1.py is in the same directory
    
    # Execute Part 1(a) and Part 1(b) to obtain C_best and data splits
    C_best, accuracy_test, conf_matrix, X_train, X_val, X_test, y_train, y_val, y_test = q1.q1_main()
    
    # Execute Part 1(c) using the best C and existing data splits
    best_degree = q1c_main(C_best, X_train, y_train, X_val, y_val, X_test, y_test)
