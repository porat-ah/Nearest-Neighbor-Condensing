import unittest

import numpy as np

from nnc.Metric import Metric


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.x1 = np.array([1, 2, 3])
        self.x2 = np.array([4, 5, 6])
        self.x3 = self.x1.copy()

    def test_euclidean(self):
        metric = "euclidean"
        dist_function = Metric(metric)
        self.assertEqual(27**0.5, dist_function(self.x1, self.x2))
        self.assertEqual(0, dist_function(self.x1, self.x3))

    def test_manhattan(self):
        metric = "manhattan"
        dist_function = Metric(metric)
        self.assertEqual(9, dist_function(self.x1, self.x2))
        self.assertEqual(0, dist_function(self.x1, self.x3))

    def test_chebyshev(self):
        metric = "chebyshev"
        dist_function = Metric(metric)
        self.assertEqual(3, dist_function(self.x1, self.x2))
        self.assertEqual(0, dist_function(self.x1, self.x3))

    def test_minkowski(self):
        metric = "minkowski"
        dist_function = Metric(metric , p= 1)
        dist_function2 = Metric("manhattan")
        self.assertEqual(dist_function2(self.x1, self.x2), dist_function(self.x1, self.x2))
        self.assertEqual(0, dist_function(self.x1, self.x3))
        dist_function = Metric(metric, p=2)
        dist_function3 = Metric("euclidean")
        self.assertEqual(dist_function3(self.x1, self.x2), dist_function(self.x1, self.x2))
        self.assertEqual(0, dist_function(self.x1, self.x3))


if __name__ == '__main__':
    unittest.main()
