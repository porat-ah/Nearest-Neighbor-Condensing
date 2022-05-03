import unittest

import numpy as np

from NNC import NNC, Metric
from sklearn.datasets import make_classification
from sklearn.preprocessing import minmax_scale
import warnings

warnings.filterwarnings('error')


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.delta = 0.001
        self.all_data_limit = 4

    def test_fit_euclidean(self):
        metric = "euclidean"
        self.fit_test_all_data(metric)

    def test_fit_manhattan(self):
        metric = "manhattan"
        self.fit_test_all_data(metric)

    def test_fit_chebyshev(self):
        metric = "chebyshev"
        self.fit_test_all_data(metric)

    def test_transform_euclidean_brute(self):
        metric = "euclidean"
        algorithm = "brute"
        self.transform_test_all_data(algorithm, metric)

    def test_transform_euclidean_prune(self):
        metric = "euclidean"
        algorithm = "prune"
        self.transform_test_all_data(algorithm, metric)

    def fit_test_all_data(self, metric):
        for i in range(self.all_data_limit):
            self.data_type(i)
            self.fit_test(metric)
            self.data_type(i, False)
            self.fit_test(metric)

    def fit_test(self, metric):
        self.x1, self.x2, margin = self.find_margin_2_classes(self.X, self.y, metric)
        nnc = NNC(metric=metric)
        nnc.fit(X=self.X, y=self.y)
        self.assertAlmostEqual(margin, nnc.gamma, delta=self.delta)

    def transform_test_all_data(self, algorithm, metric):
        for i in range( self.all_data_limit):
            self.data_type(i)
            self.transform_test(algorithm, metric)
            self.data_type(i, False)
            self.transform_test(algorithm, metric)

    def transform_test(self, algorithm, metric):
        dist_function = Metric(metric)
        nnc = NNC(algorithm=algorithm, metric=metric)
        nnc.fit(X=self.X, y=self.y)
        X_new, y_new = nnc.transform(self.X, self.y)
        X_new[:, 0] = (X_new[:, 0] - self.X[:, 0].min()) / (self.X[:, 0].max() - self.X[:, 0].min())
        X_new[:, 1] = (X_new[:, 1] - self.X[:, 1].min()) / (self.X[:, 1].max() - self.X[:, 1].min())
        self.X = minmax_scale(self.X, feature_range=(0, 1))
        scale = dist_function(np.ones_like(self.X[0]), np.zeros_like(self.X[0]))
        for i, x in enumerate(self.X):
            margin = 1
            index = 0
            for j, x_ in enumerate(X_new):
                _margin = dist_function(x, x_) / scale
                if margin > _margin:
                    margin = _margin
                    index = j
            self.assertEqual(self.y[i], y_new[index])

    def find_margin_2_classes(self, X, y, metric):
        X = minmax_scale(X, feature_range=(0, 1))
        dist_function = Metric(metric)
        margin = 1
        X1 = X[y == np.unique(y)[0]]
        X2 = X[y == np.unique(y)[1]]
        x1 = x2 = None
        scale = dist_function(np.ones_like(X[0]), np.zeros_like(X[0]))
        for p1 in X1:
            for p2 in X2:
                _margin = dist_function(p1, p2) / scale
                if margin > _margin:
                    margin = _margin
                    x1 = p1
                    x2 = p2
        return x1, x2, margin

    def data_type(self, data_num, scale=True):
        if data_num == 0:
            self.X = np.array([
                [4, 1],
                [3, 1],
                [2, 1],
                [2.5, 0.5],
                [3.5, 0.5],
                [4, 2],
                [3, 2],
                [2, 2],
                [2.5, 2.5],
                [3.5, 2.5]
            ])

            self.y = np.ones(self.X.shape[0])
            self.y[:5] = self.y[:5] * 0

        elif data_num == 1:
            self.X, self.y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=2,
                                                 random_state=30,
                                                 flip_y=0, shuffle=False, class_sep=2)
        elif data_num == 2:
            self.X, self.y = make_classification(n_samples=2000, n_features=2, n_redundant=0, n_clusters_per_class=2,
                                                 random_state=25,
                                                 flip_y=0, shuffle=False, class_sep=1.5)
        elif data_num == 3:
            self.X, self.y = make_classification(n_samples=2000, n_features=2, n_redundant=0, n_clusters_per_class=1,
                                                 random_state=25,
                                                 flip_y=0, shuffle=False, class_sep=0.5)
        if scale:
            self.X = minmax_scale(self.X, feature_range=(0, 1))
        self.meta_data = (data_num, scale)


if __name__ == '__main__':
    unittest.main()
