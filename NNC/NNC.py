from .Metric import *
import numpy as np
from numpy import floor, log
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import pairwise_distances


class NNC:
    def __init__(self, algorithm="brute", metric="minkowski", p= 2, n_jobs=1):
        """
        :param algorithm:{"brute" , "prune"} Algorithm used to compute the cardinality subset.
        :param metric:{"euclidean", "manhattan", "minkowski", "chebyshev"} or a callable function. Distance metric to use for finding gamma.
        The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
        :param p: Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1),
        and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        """
        self.gamma = 1
        self.S_gamma = list()
        self.algorithm = algorithm
        if callable(metric):
            self.dist = metric
        else:
            self.dist = Metric(metric, p)
        self.x1 = None
        self.x2 = None
        self.scale = None
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        find gamma

        :param X: numpy array , Data
        :param y: numpy array , Target values
        :return: self
        """
        X = minmax_scale(X, feature_range=(0,1))
        self.scale = self.dist(np.ones_like(X[0]), np.zeros_like(X[0]))
        groups = []
        for label in np.unique(y):
            groups.append(X[y == label])
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                i_j_distances = np.min(
                    pairwise_distances(groups[i], groups[j], metric=self.dist, n_jobs=self.n_jobs) / self.scale)
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
        _X = minmax_scale(X, feature_range=(0, 1))
        X_new = self.brute(_X)
        mask = np.array(self.find_common_arrays_location(_X, X_new))
        y_new = np.zeros_like(X_new[:, 0])
        y_new[mask[:, 1]] = y[mask[:, 0]]
        if self.algorithm == "prune":
            X_new = self.prune(X_new, y_new)
        mask = np.array(self.find_common_arrays_location(_X, X_new))
        return X[mask[:, 0]], y[mask[:, 0]]

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        """
        run both fit and transform

        :param X: numpy array , Data
        :param y: numpy array , Target values
        :return: (numpy array , numpy array) : (X_new - Cardinality subset of X , y_new - Target values of X_new)
        """
        self.fit(X, y)
        self.transform(X, y)

    def brute(self, X: np.ndarray, y=None):
        self.S_gamma.append(X[0])
        for p in X:
            b = True
            for q in self.S_gamma:
                b &= (self.dist(p, q)/self.scale >= self.gamma)
                if not b:
                    break
            if b:
                self.S_gamma.append(p)
        return np.unique(np.array(self.S_gamma), axis=0)

    def prune(self, X: np.ndarray, y: np.ndarray) -> np.array:
        S_gama = X.copy()
        pruning_levels = np.flip(range(floor(log(self.gamma)).astype(int), 2))
        for i in pruning_levels:
            for j, p in enumerate(X):
                b = True
                for q in X[y != y[j]]:
                    b &= self.dist(p, q)/self.scale >= np.power(2., i + 1)
                    if not b:
                        break
                if b:
                    for p_ in X:
                        if (np.any(p != p_)) and (self.dist(p, p_)/self.scale < np.power(2.0, i) - self.gamma):
                            np.delete(S_gama, np.where(np.atleast_1d(np.all(S_gama == p_)).nonzero()))

        return S_gama


    def find_common_arrays_location(self, arr1, arr2):
        index = []
        for i, a in enumerate(arr1):
            for j, b in enumerate(arr2):
                if np.all(a == b):
                    index.append([i, j])
                    break
        return index

    def __str__(self):
        return "NNC(algorithm='{alg}', metric='{mtr}', p={p})".format(alg=self.algorithm,
                                                                     mtr=self.dist.metric,
                                                                     p=self.dist.p)
