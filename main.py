from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x , y = make_classification(n_samples= 1000 , n_features= 2 , n_redundant= 0 , n_clusters_per_class= 2 , random_state = 30 , flip_y = 0 , shuffle= False , class_sep = 2)
    x = (x - x.min())
    x = x/x.max()
    print(x)
    x1 = np.array([0.30923434, 0.29612874])
    x2 = np.array([0.33526732, 0.44080387])
    _x = np.array([x1,x2])
    print(_x)
    sns.scatterplot(x= x[: , 0] , y= x[: , 1] , hue= y)
    plt.scatter(x= _x[: , 0] , y= _x[: , 1] , color= "r")
    plt.show()