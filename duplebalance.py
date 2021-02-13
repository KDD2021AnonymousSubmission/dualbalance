# -*- coding: utf-8 -*-
"""
@author: KDD2021AnonymousSubmission
mailto: anonymous.submit@foxmail.com
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_random_state, check_is_fitted, column_or_1d, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier  
from utils import (
    undersample_single_class,
    oversample_single_class,
    macro_auc_roc_score,
)
from collections import Counter
from base import _partition_estimators, delayed, _parallel_predict_proba
from joblib import Parallel

class DupleBalanceClassifier(BaseEnsemble):
    """A DupleBalance ensemble classifier for imbalanced multi-class classification.

    A DupleBalance classifier is an ensemble meta-estimator that fits base 
    classifiers each on a resampled and augmented dataset for achieving 
    inter-class and intra-class balancing. Specifically, DupleBalance 
    achieves inter-class balancing via progressive hybrid sampling (i.e.,
    undersample the majority and oversample the minority classes),
    and intra-class balancing by computing and harmonizing the prediction 
    error distribution.

    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeClassifier`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    n_bins : int, default=5
        The number of bins in the histogram (for approximation 
        of the error distribution).

    alpha : float, defalut=0
        The perturbation coefficient (for adjustment of the 
        intensity of data augmentation).

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    n_jobs : int, default=None
        The number of jobs to run in parallel for :meth:`predict`. 
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` 
        context. ``-1`` means using all processors. See 
        :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    n_features_ : int
        The number of features when :meth:`fit` is performed.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int or list
        The number of classes.
        
    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier 
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = DupleBalanceClassifier(base_estimator=DecisionTreeClassifier(),
    ...                    n_estimators=10, random_state=0).fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])

    """
    def __init__(self, 
            base_estimator = DecisionTreeClassifier(), 
            n_estimators: int = 10, 
            n_bins: int = 5,
            alpha: float = 0,
            estimator_params = tuple(),
            n_jobs = None,
            random_state = None,
            verbose = 0,
            ):
        
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_bins = n_bins
        self.alpha = alpha
        self.estimator_params = estimator_params
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def compute_pbhs_distribution(self, current_iter: int, total_iter: int) -> dict:
        """Progressively-balanced hybrid sampling (PBHS). Compute the expected 
           class distribution (number of samples from each class) after PBHS.

        Parameters
        ----------
        current_iter : int
            The current number of iterations of the ensemble training.
        
        total_iter : int
            The total number of iterations of the ensemble training.

        Returns
        -------
        pbhs_class_distr : dict of shape (n_classes,)
            Keys: int, the class labels. 
            Values: int, the expected cardinality of the corresponding 
                class after PBHS.
        """

        org_class_num = np.array(list(self.org_class_distr.values()))
        progress = min(current_iter / total_iter, 1)
        bal_class_num = np.full_like(org_class_num, self.n_ave)

        pbhs_class_num = np.round(
            bal_class_num * progress + org_class_num * (1-progress)
            ).astype(int).tolist()
        pbhs_class_distr = dict(zip(self.org_class_distr.keys(), pbhs_class_num))
        self.pbhs_class_distribution = pbhs_class_distr

        return pbhs_class_distr
    
    def compute_inverse_error_distribution_weights(self, y_true, y_pred_proba):
        """Compute the intra-class balanced sampling weights by 
           inversing prediction error distribution.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            The ground-truth target values (class labels in classification, real numbers in
            regression).
        
        y_pred_proba : array-like of shape (n_samples, n_features)
            The prediction probabilities of samples in X.

        Returns
        -------
        inverse_weights : array-like of shape (n_samples,)
            The intra-class balanced sampling weights. 
        """

        if y_true.shape != y_pred_proba.shape:
            one_hot_inds = label_binarize(y_true, classes=self.classes_).astype(bool)
            if one_hot_inds.shape[1] == 1:
                one_hot_inds = np.concatenate([one_hot_inds, ~one_hot_inds], axis=1)
            y_pred_proba = y_pred_proba[one_hot_inds]

        n_bins, previous_num_in_bin = self.n_bins, self.error_distribution
        errors = np.abs(y_pred_proba - np.ones_like(y_pred_proba))
        inverse_weights = np.ones_like(errors)
        num_in_bin_list = []
        total_num = len(y_true)
        momentum = 0.1
        edges = np.arange(n_bins + 1) / n_bins
        edges[-1] += 1e-6

        n = 0
        for i in range(n_bins):
            inds = (errors >= edges[i]) & (errors < edges[i+1])
            num_in_bin = inds.sum()
            if previous_num_in_bin is not None:
                num_in_bin = num_in_bin * (1-momentum) + previous_num_in_bin[i] * momentum
            num_in_bin_list.append(num_in_bin)
            if num_in_bin > 0:
                inverse_weights[inds] = total_num / num_in_bin
                n += 1

        if n > 0:
            inverse_weights /= n
        
        self.error_distribution = num_in_bin_list

        return inverse_weights
    
    def resample(self, X, y, y_pred_proba, current_iter: int, total_iter: int):
        """Resample the training data by inter-class and intra-class balancing.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        
        y_pred_proba : array-like of shape (n_samples, n_features)
            The prediction probabilities of samples in X.
        
        current_iter : int
            The current number of iterations of the ensemble training.
        
        total_iter : int
            The total number of iterations of the ensemble training.

        Returns
        -------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The resampled training feature vectors. 

        y : array-like of shape (n_samples,)
            The resampled training labels. 
        """
        pbhs_class_distr = self.compute_pbhs_distribution(current_iter, total_iter)
        inverse_weights = self.compute_inverse_error_distribution_weights(y, y_pred_proba)
        inverse_weights = np.clip(inverse_weights, 0, 10)

        X_res_list, y_res_list = [], []
        for label in pbhs_class_distr.keys():
            idx = (y == label)
            n, n_expect = idx.sum(), pbhs_class_distr[label]
            label_weights = inverse_weights[idx]
            if n >= n_expect:
                X_res, y_res = undersample_single_class(
                    X[idx], label, n_expect, label_weights, self.random_state)
            else:
                X_res, y_res = oversample_single_class(
                    X[idx], label, n_expect, label_weights, self.random_state)
            X_res_list.append(X_res)
            y_res_list.append(y_res)

        return np.concatenate(X_res_list), np.concatenate(y_res_list), inverse_weights
    
    def compute_base_class_statistic(self, X, y):
        """Compute the statistics of all base classes for perturbation-based
           data augmentation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        base_class_statistics : dict of shape (n_classes,)
            Keys: int, the class labels. 
            Values: dict, the corresponding statistics.
        """

        base_class_statistics = {}
        for label in self.classes_:
            idx = (y == label)
            X_i = X[idx]
            mean = np.mean(X_i, axis=0)
            std = np.std(X_i, axis=0)
            cov = np.cov(X_i.T)
            base_class_statistics[label] = {
                'mean': mean, 'std': std, 'cov': cov, 'count': idx.sum()
                }
        return base_class_statistics
    
    def perturbation_data_augment(self, X, y):
        """Add perturbation signal to the feature vector for data augmentation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples after data augmentation. 

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        """
        X_perturb = np.zeros_like(X)
        for label in np.unique(y):
            stat = self.base_class_statistics[label]
            idx = (y==label)
            X_perturb_label = np.random.multivariate_normal(
                mean=np.zeros(self.n_features_), 
                cov=stat['cov'], 
                size=idx.sum())
            X_perturb_label *= self.alpha
            X_perturb[idx] = X_perturb_label

        return X + X_perturb, y

    def _validate_y(self, y):
        """Validate the label vector."""

        y = column_or_1d(y, warn=True)
        check_classification_targets(y)

        return y
    
    def update_pred_buffer(self, X):
        """Maintain a latest prediction probabilities of the training data 
           during ensemble training."""

        if self.n_buffered_estimators_ > len(self.estimators_):
            raise ValueError(
                'Number of buffered estimators ({}) > total estimators ({}), check usage!'.format(
                    self.n_buffered_estimators_, len(self.estimators_)))
        if self.n_buffered_estimators_ == 0:
            self.y_pred_proba_buffer = np.full(shape=(self._n_samples, self.n_classes_), fill_value=1./self.n_classes_)
        y_pred_proba_buffer = self.y_pred_proba_buffer
        for i in range(self.n_buffered_estimators_, len(self.estimators_)):
            y_pred_proba_i = self.estimators_[i].predict_proba(X)
            y_pred_proba_buffer = (y_pred_proba_buffer * i + y_pred_proba_i) / (i+1)
        self.y_pred_proba_buffer = y_pred_proba_buffer
        self.n_buffered_estimators_ = len(self.estimators_)

        return
    
    def init_data_statistics(self, X, y, to_console=False):
        """Initialize DupleBalance with training data statistics."""
        self._n_samples, self.n_features_ = X.shape
        self.features_ = np.arange(self.n_features_)
        self.org_class_distr = Counter(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_ave = self._n_samples / self.n_classes_
        self.n_buffered_estimators_ = 0
        self.error_distribution = None
        self.base_class_statistics = self.compute_base_class_statistic(X, y)
        if to_console:
            print ('----------------------------------------------------')
            print ('# Samples     : {}'.format(self._n_samples))
            print ('# Features    : {}'.format(self.features_))
            print ('# Classes     : {}'.format(self.n_classes_))
            cls_label, cls_dis, IRs = '', '', ''
            min_n_samples = min(self.org_class_distr.values())
            for label, num in sorted(self.org_class_distr.items(), key=lambda d: d[1], reverse=True):
                cls_label += f'{label}/'
                cls_dis += f'{num}/'
                IRs += '{:.2f}/'.format(num/min_n_samples)
            print ('Classes       : {}'.format(cls_label[:-1]))
            print ('Class Dist    : {}'.format(cls_dis[:-1]))
            print ('McIR          : {}'.format(IRs[:-1]))
            print ('----------------------------------------------------')
        pass

    def fit(self, X, y):
        """Build a DupleBalance ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
        """
        
        # validate data format and estimator
        self.random_state = check_random_state(self.random_state)
        X, y = self._validate_data(
            X, y, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False, multi_output=True)
        y = self._validate_y(y)
        self._validate_estimator()

        # initialization
        self.init_data_statistics(X, y, to_console=True if self.verbose > 0 else False)
        self.estimators_ = []
        self.estimators_features_ = []
        
        for i_iter in range(1, self.n_estimators+1):

            # update current training data prediction
            self.update_pred_buffer(X)

            # perform inter-class and intra-class balanced sampling
            X_resampled, y_resampled, weights = self.resample(
                X.copy(), y.copy(), self.y_pred_proba_buffer, i_iter, self.n_estimators)
            
            if self.verbose > 0:
                print ('Iteration {:<4d}: training set class distribution {}'.format(
                    i_iter, self.pbhs_class_distribution,))
            
            # data augmentation
            X_augmented, y_augmented = self.perturbation_data_augment(
                X_resampled, y_resampled)

            # train a new base estimator and add it into self.estimators_
            estimator = self._make_estimator(append=True, random_state=self.random_state)
            estimator.fit(X_augmented, y_augmented)
            self.estimators_features_.append(self.features_)
        
        return self

    def _parallel_args(self):
        return {}
        
    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = check_array(
            X, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False
        )
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))
        
        # return np.array(
        #     [model.predict_proba(X) for model in self.estimators_]
        #     ).mean(axis=0)
        
        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             **self._parallel_args())(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

# test example
if __name__ == '__main__':

    from sklearn.datasets import make_classification
    from utils import make_long_tail, imbalance_train_test_split
    import argparse

    # prepare dataset
    X, y = make_classification(n_samples=1000, n_features=4,  
                            n_classes=3, class_sep=0.8,
                            n_informative=3, n_redundant=1,
                            random_state=42, shuffle=False)
    X, y = make_long_tail(X, y, imb_type='log', log_n=2, imb_ratio=10, random_state=42)
    X_train, X_test, y_train, y_test = imbalance_train_test_split(X, y, test_size=0.5, random_state=0)

    # argument parser
    parser = argparse.ArgumentParser(description='DupleBalance Arguments')
    parser.add_argument('--n_estimators', type=int, default=10,
                    help='The number of base estimators in the ensemble. (default: 10)')
    parser.add_argument('--n_bins', type=int, default=5,
                    help='The number of bins in the histogram (for approximation \
                    of the error distribution). (default: 5)')
    parser.add_argument('--alpha', type=float, default=0.,
                    help='The perturbation coefficient (for adjustment of the \
                    intensity of data augmentation). (default: 0.)')        
    parser.add_argument('--n_jobs', type=int, default=None,
                    help='The number of jobs to run in parallel for :meth:`predict`. \
                    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` \
                    context. ``-1`` means using all processors. See \
                    :term:`Glossary <n_jobs>` for more details. (default: None)')
    parser.add_argument('--verbose', type=int, default=1,
                    help='Controls the verbosity when fitting and predicting. \
                    ``0`` means no output to console. ``1`` means print message \
                    to console when fitting and predicting. (default: 1)')        
    
    # initialize DupleBalanceClassifier instance
    args = parser.parse_args()
    clf = DupleBalanceClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=args.n_estimators, 
        n_bins=args.n_bins,
        alpha=args.alpha,
        n_jobs=args.n_jobs,
        verbose=args.verbose,)
    
    # fit and predict
    print ('\nA minimal working example:')
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)
    score = macro_auc_roc_score(y_test, y_pred_proba)
    print (f'\nMacro AUROC score: {score}\n')