from sklearn.datasets import make_classification
from sklearn.preprocessing import minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from NNC import NNC

if __name__ == '__main__':
    x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=2, random_state=30,
                               flip_y=0, shuffle=False, class_sep=2)
    #nnc = NNC("chebyshev")
    #nnc.fit(x, y)
    x = minmax_scale(x , feature_range=(0,1))
    #_x = np.array([nnc.x1, nnc.x2])
    _x = np.array([[0.29858597,0.29612874], [0.32533451,0.44080387]])
    _x = np.array([[0.20764637, 0.32039697],[0.32533451,0.44080387]])

    #print(nnc.gama)
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y)
    plt.scatter(x=_x[:, 0], y=_x[:, 1], color="r")
    plt.show()
