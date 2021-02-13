# -*- coding: utf-8 -*-
"""
@author: KDD2021AnonymousSubmission
mailto: anonymous.submit@foxmail.com
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def imbalance_train_test_split(X, y, test_size, random_state=None):
    '''Train/Test split that guarantee same class distribution between split datasets.'''
    classes = np.unique(y)
    X_trains, y_trains, X_tests, y_tests = [], [], [], []
    for label in classes:
        inds = (y==label)
        X_label, y_label = X[inds], y[inds]
        X_train, X_test, y_train, y_test = train_test_split(
            X_label, y_label, test_size=test_size, random_state=random_state)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    X_train = np.concatenate(X_trains)
    X_test = np.concatenate(X_tests)
    y_train = np.concatenate(y_trains)
    y_test = np.concatenate(y_tests)
    return  X_train, X_test, y_train, y_test

def oversample_single_class(X, label, n_expect, weights, random_state):
    X = pd.DataFrame(X).sample(
        n=n_expect, weights=weights, replace=True, random_state=random_state
        ).values
    y = np.full(X.shape[0], label)
    return X, y

def undersample_single_class(X, label, n_expect, weights, random_state):
    if X.shape[0] >= n_expect:
        X = pd.DataFrame(X).sample(
            n=n_expect, weights=weights, replace=False, random_state=random_state
        ).values
    else:
        raise ValueError('Class: {} #sample: {} < {}'.format(label, X.shape[0], n_expect))
    y = np.full(X.shape[0], label)
    return X, y

def macro_auc_roc_score(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovo')

def make_long_tail(X, y, imb_type='log', log_n=2, imb_ratio=10, random_state=42):
    count_sorted = np.array(sorted(Counter(y).items(), key=lambda d: d[1], reverse=True))
    classes_sorted, class_num_sorted = count_sorted[:, 0], count_sorted[:, 1]
    num_class = len(count_sorted)
    num_head = class_num_sorted[0]
    num_tail = int(num_head / imb_ratio)
    num_gap  = num_head - num_tail
    if imb_type == 'log':
        class_num_expect = [int(num_tail+weight*num_gap) for weight in np.power(np.linspace(1, 0, num_class), log_n)] 
    elif imb_type == 'linear':
        class_num_expect = [int(num_head*(1-i/(num_class-1))+num_tail*i/(num_class-1)) for i in range(num_class)]
    else:
        raise ValueError('imb_type {} is not supported, try ["log"/"linear"].'.format(imb_type))
    X_sampled, y_sampled = [], []
    for class_i, class_num in zip(classes_sorted, class_num_expect):
        X_i = X[y==class_i]
        if X_i.shape[0] > class_num:
            X_i = pd.DataFrame(X_i).sample(n=class_num, 
                random_state=random_state, replace=False).values
        y_i = np.full(X_i.shape[0], class_i)
        X_sampled.append(X_i)
        y_sampled.append(y_i)
    X_longtail = np.concatenate(X_sampled)
    y_longtail = np.concatenate(y_sampled)
    return X_longtail, y_longtail