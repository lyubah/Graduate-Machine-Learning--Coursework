# -*- coding: utf-8 -*-

import numpy as np


class NaiveBayesClassifier:
    def __init__(self, alpha):
        self.alpha = alpha   
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        # Initialize parameters
        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            # Apply Laplace smoothing
            smoothed_count = X_c.sum(axis=0) + self.alpha
            smoothed_total = smoothed_count.sum()
            self.feature_log_prob_[idx, :] = np.log(smoothed_count / smoothed_total)

    def predict(self, X):
        return np.array([self._predict_single(sample) for sample in X])

    def _predict_single(self, x):
        log_probs = self.class_log_prior_ + x @ self.feature_log_prob_.T
        return self.classes[np.argmax(log_probs)]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)