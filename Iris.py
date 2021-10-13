import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

iris_target = iris.target.reshape((150, 1))
iris_data = np.hstack((iris.data, iris_target))
iris_df = pd.DataFrame(iris_data)
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print(iris_df.head())


# 鸢尾花
def Iris():
    # 特征间相关性
    sns.pairplot(iris_df, hue='species', markers=['s', 'o', '^'])
    plt.show()

    # 使用热力图可视化相关性
    iris_corr = iris_df.drop('species', axis=1).corr()
    sns.heatmap(iris_corr)
    plt.show()


# 线性可分
def Iris_LSVC():
    sub_data1 = iris_df[iris_df.species != 2]
    sub_X1 = sub_data1.drop(['species', 'sepal_length', 'sepal_width'], axis=1)
    sub_Y1 = sub_data1.species

    clf1 = svm.LinearSVC()
    clf1.fit(sub_X1, sub_Y1)

    # 获取分类结果
    prediction = clf1.predict(sub_X1)

    # 绘制分类超平面
    w = clf1.coef_
    b = clf1.intercept_

    x = np.linspace(1.5, 3.5, 100)
    y = (w[0, 0] * x + b[0]) / (-w[0, 1])
    plt.plot(x, y, 'r')

    c1 = sub_Y1 == 0
    c2 = sub_Y1 == 1
    colors = np.asarray([i for i in map(lambda a: 'yellowgreen' if a == 1 else 'steelblue', prediction)])

    print(sub_Y1 == 0)

    # print(c1)
    # print(c2)

    print(sub_X1)

    # plt.scatter(sub_X1[c1, 0], sub_X1[c1, 1], c=colors[c1], s=60, alpha=0.5, marker='s')
    # plt.scatter(sub_X1[c2, 0], sub_X1[c2, 1], c=colors[c2], s=60, alpha=0.5, marker='o')

    plt.grid()
    plt.show()
