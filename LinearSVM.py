import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def LSVM():
    # 生成数据
    centers = [(-1, -0.125), (0.5, 0.5)]
    X, y = make_blobs(n_samples=50, n_features=2, centers=centers, cluster_std=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # print(X.shape)
    # print(y.shape)
    # print(X)
    # print(y)

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # print(X_train)
    # print(y_train)

    # df_x = pd.DataFrame(X[:, 0], columns=['x'])
    # df_y = pd.DataFrame(X[:, 1], columns=['y'])
    # df_tag = pd.DataFrame(y, columns=['tag'])
    # df = pd.concat([df_x, df_y, df_tag], axis=1)
    # print(df.head())

    # sns.lmplot(x='x', y='y', hue='tag', data=df,
    #            height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
    # plt.show()

    model = LinearSVC()
    model.fit(X_train, y_train)  # 训练
    # a = model.coef_
    # b = model.intercept_
    # print(a)
    # print(b)
    # x=np.linspace(-2,1.5,100)
    # plt.plot(x,a[:, 1]+a[:, 0]*x)
    # plt.show()

    y_pred = model.predict(X_test)
    print(y_pred)
    accuracy_score(y_pred, y_test)  # 评估
    print(accuracy_score(y_pred, y_test))
