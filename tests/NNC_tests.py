import unittest

import numpy as np

from NNC import NNC , Metric
from sklearn.datasets import make_classification


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=2, random_state=30,
                                   flip_y=0, shuffle=False, class_sep=2)
        x = (x - x.min())
        x = x / x.max()
        self.x = x
        self.y = y
        self.delta = 0.001
        self.x1 = np.array([0.30923434, 0.29612874])
        self.x2 = np.array([0.33526732, 0.44080387])

    def test_fit_euclidean(self):
        metric = "euclidean"
        self.fit_test(metric)

    def test_fit_manhattan(self):
        metric = "manhattan"
        self.fit_test(metric)

    def test_fit_chebyshev(self):
        metric = "chebyshev"
        self.fit_test(metric)

    def fit_test(self , metric):
        dist_function = Metric(metric)
        nnc = NNC(metric=metric)
        nnc.fit(X=self.x, y=self.y)
        self.assertAlmostEqual(dist_function(self.x1, self.x2), nnc.gama, delta=self.delta)


if __name__ == '__main__':
    unittest.main()
