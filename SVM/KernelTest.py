import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def plot_dataset(X, y):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "s")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "o")
    plt.grid(True, which="both")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")


def plot_predict(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap=plt.cm.winter, alpha=0.5)
    plt.contour(x0, x1, y_decision, cmap=plt.cm.winter, alpha=0.2)


def KernelTest():
    X, y = make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=200)
    # X, y = make_moons(n_samples=200,noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    df_x = pd.DataFrame(X[:, 0], columns=['x'])
    df_y = pd.DataFrame(X[:, 1], columns=['y'])
    df_tag = pd.DataFrame(y, columns=['tag'])
    df = pd.concat([df_x, df_y, df_tag], axis=1)

    sns.lmplot(x='x', y='y', hue='tag', data=df,
               height=6, fit_reg=False, scatter_kws={"s": 50})
    plt.show()

    kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
    for i in kernel_list:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel=i, gamma=10))
        ])
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print(i + ':' + str(accuracy_score(prediction, y_test)))

        plot_dataset(X_test, y_test)
        plot_predict(model, [-3, 3, -3, 3])
        plt.title(i)
        plt.show()
