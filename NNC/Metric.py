import numpy as np


class Metric:

    metrics = ["euclidean", "manhattan", "minkowski", "chebyshev"]

    def __init__(self, metric="minkowski", p=2):
        pMap = {"euclidean": 2, "manhattan": 1, "minkowski": p}
        self.metric = metric
        self.p = pMap.get(metric)
        if metric == "chebyshev":
            def dist(X1: np.ndarray, X2: np.ndarray) -> float:
                return np.max(np.abs(X1 - X2))

            self.func = dist
        else:
            self.func = self.distFunc()

    def distFunc(self):
        def dist(X1: np.ndarray, X2: np.ndarray) -> float:
            return np.power(np.sum(np.power(np.abs(X1 - X2), self.p)), 1 / self.p)

        return dist

    def __call__(self, X1: np.ndarray, X2: np.ndarray):
        return self.func(X1, X2)
