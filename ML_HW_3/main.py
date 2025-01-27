# -*- coding: utf-8 -*-

from utility import * 
from Naive_Bayes import *


def main():

    data_dir = "fortune-cookie-data/"
    stop_words_file = data_dir + "stoplist.txt"
    train_texts_file = data_dir + "traindata.txt"
    train_labels_file = data_dir + "trainlabels.txt"
    test_texts_file = data_dir + "testdata.txt"
    test_labels_file = data_dir + "testlabels.txt"
    stop_words = load_stop_words(stop_words_file)
    train_texts_raw = read_lines(train_texts_file)
    test_texts_raw = read_lines(test_texts_file)
    y_train = np.array([int(label) for label in read_lines(train_labels_file)])
    y_test = np.array([int(label) for label in read_lines(test_labels_file)])
    train_tokens = tokenize_and_remove_stopwords(train_texts_raw, stop_words)
    test_tokens = tokenize_and_remove_stopwords(test_texts_raw, stop_words)
    vocabulary = build_vocab(train_tokens)
    X_train = texts_to_binary_matrix(train_tokens, vocabulary)
    X_test = texts_to_binary_matrix(test_tokens, vocabulary)

    nb_classifier = NaiveBayesClassifier(alpha=1)
    nb_classifier.fit(X_train, y_train)

    # Evaluate on training data
    train_accuracy = nb_classifier.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    # Evaluate on testing data
    test_accuracy = nb_classifier.score(X_test, y_test)
    print(f"Testing Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
