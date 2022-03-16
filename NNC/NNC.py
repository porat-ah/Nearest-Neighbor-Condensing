from .Metric import *
import numpy as np


class NNC:
    def __init__(self, algorithm="brute", metric="minkowski", p=2):
        self.gama = 1
        self.S_gama = set()
        self.algorithm = algorithm
        self.dist = Metric(metric=metric, p=p)

    def fit(self, X: np.ndarray, y: np.ndarray):
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

    def transform(self, X: np.ndarray, y=None):
        pass

    def fit_transfrom(self, X: np.ndarray, y: np.ndarray):
        pass

    def brute(self, X: np.ndarray):
        self.S_gama.add(X[0])
        for p in X:
            b = True
            for q in self.S_gama:
                b &= (self.dist(p, q) >= self.gama)
            if b:
                self.S_gama.add(p)
        return np.array(self.S_gama)
