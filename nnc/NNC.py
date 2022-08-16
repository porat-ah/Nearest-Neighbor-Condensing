from .Metric import *
import numpy as np
from numpy import floor, log
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm


class NNC:
    EPS = 0.00001

    def __init__(self, algorithm="brute", metric="minkowski", p=2, n_jobs=1, verbose=False):

        """
        :param algorithm: {'brute' , 'prune'},  default=’brute’
                Algorithm used to compute the cardinality subset.
        :param metric: {'euclidean', 'manhattan', 'minkowski', 'chebyshev'} or callable, default=’minkowski’
                Distance metric to use for finding gamma.
                The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
        :param p: int, default=2
                Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1),
                and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        :param n_jobs: int, default=1
                The number of jobs to use for the computation.
                This works by breaking down the pairwise matrix into n_jobs even slices and computing them in parallel.
                None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        :param verbose: boolean, default=False
                show progress bar
        """
        self.gamma = None
        self.S_gamma = list()
        self.algorithm = algorithm
        if callable(metric):
            self.dist = (True, metric, metric)
        elif metric == "minkowski":
            self.dist = (True, Metric(metric, p), metric)
        else:
            self.dist = (False, Metric(metric, p), metric)
        self.scale = None
        self.n_jobs = n_jobs
        self.verbose = not verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        find gamma

        :param X: numpy array , Data
        :param y: numpy array , Target values
        :return: self
        """
        X = np.array(X)
        y = np.array(y)
        self.scale = self.dist[1](np.full_like(X[0], np.max(X)), np.full_like(X[0], np.min(X)))
        self.gamma = self.scale
        groups = []
        for label in np.unique(y):
            groups.append(X[y == label])
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i].shape[0] > groups[j].shape[0]:
                    tmp = i
                    i = j
                    j = tmp
                for p in tqdm(groups[i], disable=self.verbose):
                    i_j_distances = self.min_pairwise_distance([p], groups[j])
                    if self.gamma > i_j_distances:
                        self.gamma = i_j_distances
        return self

    def transform(self, X: np.ndarray, y: np.ndarray):
        """
        create the cardinality subset of X using the found gamma and the chosen algorithm

        :param X: numpy array , Data
        :param y: numpy array , Target values
        :return: (numpy array , numpy array) : (X_new - Cardinality subset of X , y_new - Target values of X_new)
        """
        X = np.array(X)
        y = np.array(y)
        X_new = self.brute(X)
        mask = np.array(self.find_common_arrays_location(X, X_new))
        if self.algorithm == "prune":
            y_new = np.zeros_like(X_new[:, 0])
            y_new[mask[:, 1]] = y[mask[:, 0]]
            X_new = self.prune(X_new, y_new)
            mask = np.array(self.find_common_arrays_location(X, X_new))
        return X[mask[:, 0]], y[mask[:, 0]]

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        """
        run both fit and transform

        :param X: numpy array , Data
        :param y: numpy array , Target values
        :return: (numpy array , numpy array) : (X_new - Cardinality subset of X , y_new - Target values of X_new)
        """
        self.fit(X, y)
        return self.transform(X, y)

    def brute(self, X: np.ndarray, y=None):
        self.S_gamma.append(X[0])
        bar = tqdm(X, disable=self.verbose)
        for p in bar:
            bar.set_description("S_gamma size = {}".format(len(self.S_gamma)))
            if self.min_pairwise_distance([p], self.S_gamma) >= self.gamma - NNC.EPS:
                self.S_gamma.append(p)
        return np.unique(np.array(self.S_gamma), axis=0)

    def prune(self, X: np.ndarray, y: np.ndarray):
        S_gamma = X.copy()
        pruning_levels = np.flip(range(floor(log(self.gamma)).astype(int), 2))
        for i in tqdm(pruning_levels, disable=self.verbose):
            j = 0
            while j < len(S_gamma):
                p = S_gamma[j]
                b = True
                for q in S_gamma[y != y[j]]:
                    b &= self.dist[1](p, q) >= np.power(2., i + 1) - NNC.EPS
                    if not b:
                        break
                if b:
                    for p_ in S_gamma:
                        if (np.any(p != p_)) and (self.dist[1](p, p_) < np.power(2.0, i) - self.gamma):
                            for k, s in enumerate(S_gamma):
                                if np.all(s == p_):
                                    S_gamma = np.delete(S_gamma, k, axis=0)
                                    y = np.delete(y, k, axis=0)
                                    break
                j += 1
        return S_gamma

    def find_common_arrays_location(self, arr1, arr2):
        index = []
        for i in tqdm(range(len(arr1)), disable=self.verbose):
            for j, b in enumerate(arr2):
                if np.all(arr1[i] == b):
                    index.append([i, j])
                    break
        return index

    def min_pairwise_distance(self, g1, g2):
        if self.dist[0]:
            return np.min(
                pairwise_distances(g1, g2, metric=self.dist[1], n_jobs=self.n_jobs))
        else:
            return np.min(
                pairwise_distances(g1, g2, metric=self.dist[2], n_jobs=self.n_jobs))

    def __str__(self):
        return "NNC(algorithm='{alg}', metric='{mtr}')".format(alg=self.algorithm,
                                                               mtr=self.dist[2])
