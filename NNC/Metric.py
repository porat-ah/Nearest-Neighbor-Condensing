import numpy as np


class Metric:
    METRICS = ["euclidean", "manhattan", "minkowski", "chebyshev"]

    def __init__(self, metric="minkowski", p=2):
        """
        :param metric:{"euclidean", "manhattan", "minkowski", "chebyshev"} Distance metric to use for finding gamma.
        The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
        :param p: Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1),
        and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        """
        pMap = {"euclidean": 2, "manhattan": 1, "minkowski": p}
        self.metric = metric
        self.p = pMap.get(metric)
        if metric == "chebyshev":
            def dist(X1: np.ndarray, X2: np.ndarray) -> float:
                return np.max(np.abs(X1 - X2))

            self.func = dist
        else:
            self.func = self.dist_func()

    def dist_func(self):
        def dist(X1: np.ndarray, X2: np.ndarray) -> float:
            return np.power(np.sum(np.power(np.abs(X1 - X2), self.p)), 1 / self.p)

        return dist

    def __call__(self, X1: np.ndarray, X2: np.ndarray):
        return self.func(X1, X2)
