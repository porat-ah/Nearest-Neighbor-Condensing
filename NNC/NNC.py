from .Metric import *
import numpy as np
from sklearn.preprocessing import minmax_scale


class NNC:
    def __init__(self, algorithm="brute", metric="minkowski", p=2):
        self.gama = 1
        self.S_gama = list()
        self.algorithm = algorithm
        self.dist = Metric(metric=metric, p=p)
        self.x1 = None
        self.x2 = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = minmax_scale(X, feature_range=(0, 1))
        groups = []
        for label in np.unique(y):
            groups.append(X[y == label])
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                for x1 in groups[i]:
                    for x2 in groups[j]:
                        margin = self.dist(x1, x2)
                        if self.gama > margin:
                            self.gama = margin
                            self.x1 = x1
                            self.x2 = x2

    def transform(self, X: np.ndarray, y):
        if self.algorithm == "brute":
            X_new = self.brute(X)
        mask = np.array(self.find_common_arrays_location(X, X_new))
        y_new = np.zeros_like(X_new[:, 0])
        y_new[mask[:, 1]] = y[mask[:, 0]]
        return X_new, y_new

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        self.transform(minmax_scale(X, feature_range=(0, 1), copy=False))

    def brute(self, X: np.ndarray):
        self.S_gama.append(X[0])
        for p in X:
            b = True
            for q in self.S_gama:
                b &= (self.dist(p, q) >= self.gama)
            if b:
                self.S_gama.append(p)
        return np.unique(np.array(self.S_gama), axis=0)

    def find_common_arrays_location(self, arr1, arr2):
        index = []
        for i, a in enumerate(arr1):
            for j, b in enumerate(arr2):
                if np.all(a == b):
                    index.append([i, j])
                    break
        return index
