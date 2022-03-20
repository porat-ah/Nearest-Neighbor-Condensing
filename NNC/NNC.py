from .Metric import *
import numpy as np
from sklearn.preprocessing import minmax_scale


class NNC:
    def __init__(self, algorithm="brute", metric="minkowski", p=2):
        self.gama = 1
        self.S_gama = set()
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

    def transform(self, X: np.ndarray, y=None):
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
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
