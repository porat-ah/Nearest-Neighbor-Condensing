import unittest

import numpy as np

from NNC import NNC, Metric
from sklearn.datasets import make_classification
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.delta = 0.001

    def test_fit_euclidean(self):
        metric = "euclidean"
        self.data_type_1()
        self.fit_test(metric)
        self.data_type_2()
        self.fit_test(metric)
        self.data_type_3()
        self.fit_test(metric)
        self.data_type_1(False)
        self.fit_test(metric)
        self.data_type_2(False)
        self.fit_test(metric)
        self.data_type_3(True)
        self.fit_test(metric)

    def test_fit_manhattan(self):
        metric = "manhattan"
        self.data_type_1()
        self.fit_test(metric)
        self.data_type_2()
        self.fit_test(metric)
        self.data_type_3()
        self.fit_test(metric)
        self.data_type_1(False)
        self.fit_test(metric)
        self.data_type_2(False)
        self.fit_test(metric)
        self.data_type_3(True)
        self.fit_test(metric)

    def test_fit_chebyshev(self):
        metric = "chebyshev"
        self.data_type_1()
        self.fit_test(metric)
        self.data_type_2()
        self.fit_test(metric)
        self.data_type_3()
        self.fit_test(metric)
        self.data_type_1(False)
        self.fit_test(metric)
        self.data_type_2(False)
        self.fit_test(metric)
        self.data_type_3(True)
        self.fit_test(metric)

    def test_transform_euclidean(self):
        metric = "euclidean"
        self.data_type_1()
        self.transform_test(metric)

    def fit_test(self, metric):
        self.x1, self.x2, margin = self.find_margin_2_classes(self.X, self.y, metric)
        nnc = NNC(metric=metric)
        nnc.fit(X=self.X, y=self.y)
        self.assertAlmostEqual(margin, nnc.gama, delta=self.delta)

    def transform_test(self, metric):
        dist_function = Metric(metric)
        nnc = NNC(metric=metric)
        nnc.fit(X=self.X, y=self.y)
        X_new, y_new = nnc.transform(self.X, self.y)
        for i, x in enumerate(self.X):
            margin = 1
            index = 0
            for j, x_ in enumerate(X_new):
                if margin > dist_function(x, x_):
                    margin = dist_function(x, x_)
                    index = j
            self.assertEqual(self.y[i], y_new[index])

    def find_margin_2_classes(self, X, y, metric):
        X = minmax_scale(X, feature_range=(0, 1))
        dist_function = Metric(metric)
        margin = 1
        X1 = X[y == np.unique(y)[0]]
        X2 = X[y == np.unique(y)[1]]
        x1 = x2 = None
        for p1 in X1:
            for p2 in X2:
                if margin > dist_function(p1, p2):
                    margin = dist_function(p1, p2)
                    x1 = p1
                    x2 = p2
        return x1, x2, margin

    def data_type_1(self, scale=True):
        self.X, self.y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=2,
                                             random_state=30,
                                             flip_y=0, shuffle=False, class_sep=2)
        if scale:
            self.X = minmax_scale(self.X, feature_range=(0, 1))

    def data_type_2(self, scale=True):
        self.X, self.y = make_classification(n_samples=2000, n_features=2, n_redundant=0, n_clusters_per_class=2,
                                             random_state=25,
                                             flip_y=0, shuffle=False, class_sep=1.5)
        if scale:
            self.X = minmax_scale(self.X, feature_range=(0, 1))

    def data_type_3(self, scale=True):
        self.X, self.y = make_classification(n_samples=2000, n_features=2, n_redundant=0, n_clusters_per_class=1,
                                             random_state=25,
                                             flip_y=0, shuffle=False, class_sep=0.5)
        if scale:
            self.X = minmax_scale(self.X, feature_range=(0, 1))


if __name__ == '__main__':
    unittest.main()
