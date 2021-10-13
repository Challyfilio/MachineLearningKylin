import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def LSVM():
    # 生成数据
    centers = [(-1, -0.125), (0.5, 0.5)]
    X, y = make_blobs(n_samples=100, n_features=2, centers=centers, cluster_std=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LinearSVC()
    model.fit(X_train, y_train)  # 训练

    w = model.coef_
    b = model.intercept_
    x = np.linspace(-2, 1.5, 100)
    yl = (w[0, 0] * x + b[0]) / (-w[0, 1])

    prediction = model.predict(X_test)
    # print(prediction)

    c1 = y_test == 0
    c2 = y_test == 1
    colors = np.asarray([i for i in map(lambda a: 'yellowgreen' if a == 1 else 'steelblue', prediction)])

    plt.scatter(X_test[c1, 0], X_test[c1, 1], c=colors[c1], s=60, alpha=0.5, marker='s')
    plt.scatter(X_test[c2, 0], X_test[c2, 1], c=colors[c2], s=60, alpha=0.5, marker='o')
    plt.plot(x, yl, 'r')
    plt.show()

    print(accuracy_score(prediction, y_test))  # 评估
