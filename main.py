from sklearn.datasets import make_classification
from sklearn.preprocessing import minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from NNC import NNC


def main():
    arr1 = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    arr2 = np.array([
        [1, 2],
        [5, 6]
    ])
    print(find_common_arrays_location(arr1,arr2))

    # x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=2, random_state=30,
    #                            flip_y=0, shuffle=False, class_sep=2)
    # nnc = NNC(metric="chebyshev")
    # nnc.fit(x, y)
    # x = minmax_scale(x, feature_range=(0, 1))
    # _x = np.array([nnc.x1, nnc.x2])
    # x_new = nnc.transform(x)
    # mask = np.isin(x, x_new)
    # mask = mask[: ,0] & mask[:,1]
    # print(mask)
    # y_new = y[mask]
    # plt.figure("fig1")
    # plt.scatter(x=x_new[:, 0], y=x_new[:, 1], color="r")
    # sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y)
    # # plt.figure("fig2")
    # plt.show()


def find_common_arrays_location(arr1, arr2):
    index = []
    for i, a in enumerate(arr1):
        for j, b in enumerate(arr2):
            if np.all(a == b):
                index.append([i, j])
                break
    return index


if __name__ == '__main__':
    main()
