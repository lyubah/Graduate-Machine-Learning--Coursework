import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class KernelPerceptron:
    def __init__(self, degree=2, iterations=5, budget=1000):
        """
        Initializes the Kernel Perceptron.

        Parameters:
        - degree (int): Degree of the polynomial kernel.
        - iterations (int): Number of training iterations.
        - budget (int): Maximum number of support vectors to keep.
        """
        self.degree = degree
        self.iterations = iterations
        self.budget = budget  # Maximum number of support vectors
        self.support_vectors = []
        self.alphas = []
        self.support_labels = []
        self.classes = None
        self.mistakes_per_iteration = []

    def polynomial_kernel(self, X, Y):
        """Computes the polynomial kernel between X and Y."""
        return (1 + np.dot(X, Y.T)) ** self.degree

    def fit(self, X, y):
        """
        Trains the Kernel Perceptron.

        Parameters:
        - X (np.ndarray): Training data of shape (n_samples, n_features).
        - y (np.ndarray): Training labels of shape (n_samples,).
        """
        self.classes = np.unique(y)
        self.support_vectors = []
        self.alphas = []
        self.support_labels = []
        self.mistakes_per_iteration = []

        for it in range(1, self.iterations + 1):
            mistakes = 0
            for xi, yi in zip(X, y):
                if not self.support_vectors:
                    y_pred = None
                else:
                    K = self.polynomial_kernel(np.array([xi]), np.array(self.support_vectors))
                    # Compute scores for each class
                    scores = {}
                    for cls in self.classes:
                        idx = np.where(np.array(self.support_labels) == cls)[0]
                        if idx.size > 0:
                            scores[cls] = np.sum(np.array(self.alphas)[idx] * K[0, idx])
                        else:
                            scores[cls] = 0
                    y_pred = max(scores, key=scores.get)
                if y_pred != yi:
                    mistakes += 1
                    # Add to support vectors
                    if len(self.support_vectors) >= self.budget:
                        # Remove the oldest support vector
                        self.support_vectors.pop(0)
                        self.alphas.pop(0)
                        self.support_labels.pop(0)
                    self.support_vectors.append(xi)
                    self.alphas.append(1)
                    self.support_labels.append(yi)
            self.mistakes_per_iteration.append(mistakes)
            print(f"Iteration {it}/{self.iterations}, Mistakes: {mistakes}")
            if mistakes == 0:
                print("No mistakes in this iteration, stopping early.")
                break

    def predict(self, X):
        """
        Predicts labels for given data.

        Parameters:
        - X (np.ndarray): Data to predict of shape (n_samples, n_features).

        Returns:
        - np.ndarray: Predicted labels.
        """
        if not self.support_vectors:
            return np.array([None] * len(X))
        K = self.polynomial_kernel(X, np.array(self.support_vectors))
        scores = np.zeros((X.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            cls_indices = np.where(np.array(self.support_labels) == cls)[0]
            if cls_indices.size > 0:
                scores[:, idx] = K[:, cls_indices].dot(np.array(self.alphas)[cls_indices])
            else:
                scores[:, idx] = 0
        predictions = np.argmax(scores, axis=1)
        return self.classes[predictions]

def plot_mistakes(mistakes_per_iteration, save_path='mistakes_plot.png'):
    """
    Plots the number of mistakes per iteration.

    Parameters:
    - mistakes_per_iteration (list): Number of mistakes in each iteration.
    - save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mistakes_per_iteration) + 1), mistakes_per_iteration, marker='o', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Mistakes')
    plt.title('Mistakes per Iteration in Kernel Perceptron')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def q2_main(best_degree, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Executes Part 2: Trains Kernel Perceptron with the best polynomial degree.

    Parameters:
    - best_degree (int): Optimal polynomial degree from Part 1(c).
    - X_train, y_train (np.ndarray): Training data and labels.
    - X_val, y_val (np.ndarray): Validation data and labels.
    - X_test, y_test (np.ndarray): Testing data and labels.

    Returns:
    - tuple: (train_acc, val_acc, test_acc, mistakes_per_iteration)
    """
    # Adjust budget size as needed
    budget = 1000  # You can adjust this value based on available resources

    kp = KernelPerceptron(degree=best_degree, iterations=5, budget=budget)
    kp.fit(X_train, y_train)

    # Predict and evaluate
    y_train_pred = kp.predict(X_train)
    y_val_pred = kp.predict(X_val)
    y_test_pred = kp.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Testing Accuracy: {test_acc * 100:.2f}%")

    # Plot mistakes per iteration
    plot_mistakes(kp.mistakes_per_iteration)

    return train_acc, val_acc, test_acc, kp.mistakes_per_iteration

if __name__ == "__main__":
    """
    Executes Part 2 independently.
    Loads data, retrieves the best degree from Part 1(c), trains Kernel Perceptron, and evaluates performance.
    """
    # Load the Fashion MNIST data
    from tensorflow.keras.datasets import fashion_mnist

    # Load data
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_train_full = X_train_full.reshape(-1, 28*28) / 255.0
    X_test = X_test.reshape(-1, 28*28) / 255.0

    # Split training data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

    # Use the best degree found earlier (e.g., degree=2)
    best_degree = 2

    # Run the main function
    train_acc, val_acc, test_acc, mistakes_per_iteration = q2_main(best_degree, X_train, y_train, X_val, y_val, X_test, y_test)

