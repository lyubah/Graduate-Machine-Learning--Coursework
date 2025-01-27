#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:07:39 2024

@author: Lerberber
"""

import numpy as np

def read_lines(file_path):
    """Reads lines ret str"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def load_stop_words(file_path):
    """return as a set."""
    return set(read_lines(file_path))

def tokenize_and_remove_stopwords(texts, stop_words):
    """Tokenizes texts gets rid of stop words."""
    tokenized_texts = []
    for text in texts:
        tokens = [word for word in text.split() if word not in stop_words]
        tokenized_texts.append(tokens)
    return tokenized_texts

def build_vocab(tokenized_texts):
    """Build vocab with tokens"""
    vocab_set = set()
    for tokens in tokenized_texts:
        vocab_set.update(tokens)
    vocab_list = sorted(vocab_set)   
    vocab_dict = {word: idx for idx, word in enumerate(vocab_list)}
    return vocab_dict

def texts_to_binary_matrix(tokenized_texts, vocab_dict):
    """Converts tokenizs to a binary feature matrix."""
    num_samples = len(tokenized_texts)
    vocab_size = len(vocab_dict)
    feature_matrix = np.zeros((num_samples, vocab_size), dtype=int)
    for i, tokens in enumerate(tokenized_texts):
        for token in tokens:
            if token in vocab_dict:
                feature_matrix[i, vocab_dict[token]] = 1
    return feature_matrix

