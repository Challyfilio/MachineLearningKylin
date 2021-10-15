import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC


# 读取特征
def get_x(df):
    return np.array(df.iloc[:, :-1])


# 读取标签
def get_y(df):
    return np.array(df.iloc[:, -1])


def LogisticLSVM():
    data1 = pd.read_csv('E:\Workplace\Python\MachineLearning-ex2\ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
    print(data1.head())
    sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data1,
               height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
    # plt.show()
    x = get_x(data1)
    y = get_y(data1)

    model = LinearSVC()
    model.fit(x, y)  # 训练
    a = model.coef_
    b = model.intercept_
    print(a)
    print(b)

    x=np.linspace(25,105,100)
    plt.plot(x,a[:, 0]+a[:, 1]*x)
    plt.show()