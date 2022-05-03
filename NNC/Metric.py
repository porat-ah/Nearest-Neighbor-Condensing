import numpy as np
from scipy.spatial.distance import euclidean, chebyshev, minkowski


class Metric:


    def __init__(self, metric="minkowski", p=2):
        """
        :param metric:{"euclidean", "manhattan", "minkowski", "chebyshev"} Distance metric to use for finding gamma.
        The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
        :param p: Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1),
        and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        """
        FuncMap = {"euclidean": euclidean, "manhattan": lambda x, y: minkowski(x, y, 1),
                   "minkowski": lambda x, y: minkowski(x, y, p), "chebyshev": chebyshev}
        self.metric = metric
        self.func = FuncMap[metric]

    def __call__(self, X1: np.ndarray, X2: np.ndarray):
        return self.func(X1, X2)
