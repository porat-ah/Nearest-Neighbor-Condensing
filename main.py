from time import time
import mnist
from sklearn.utils import shuffle
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from NNC import NNC
from sklearn.metrics import confusion_matrix, classification_report


def main():
    X_train, y_train = shuffle(mnist.train_images(), mnist.train_labels())
    X_test, y_test = shuffle(mnist.test_images(), mnist.test_labels())
    X_train = X_train.reshape((60_000, 28 * 28))[y_train <= 2]
    y_train = y_train[y_train <= 2]
    X_test = X_test.reshape((10_000, 28 * 28))[y_test <= 2]
    y_test = y_test[y_test <= 2]
    start = time()
    nnc = NNC(algorithm="prune", metric="euclidean", n_jobs=-1, verbose= True)
    nnc.fit(X_train, y_train)
    print("\nNNC fit time = {:.3f}".format(time() - start))

    start = time()
    X_reduced_nnc, y_reduced_nnc = nnc.transform(X_train, y_train)
    print("\nNNC transform time = {:.3f}".format(time() - start))
    print("from size : ", X_train.shape[0], " to : ", X_reduced_nnc.shape[0])

    start = time()
    svm = SVC(C=3, degree=1)
    svm.fit(X_reduced_nnc, y_reduced_nnc)
    print("SVM fit time = {:.3f}".format(time() - start))

    start = time()
    y_pred_nnc = svm.predict(X_test)
    print("SVM predict time = {:.3f}".format(time() - start))

    print(classification_report(y_true=y_test, y_pred=y_pred_nnc))

    sns.heatmap(confusion_matrix(y_true=y_test, y_pred=y_pred_nnc), annot=True)
    plt.show()


if __name__ == '__main__':
    main()
