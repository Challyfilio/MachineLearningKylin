import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


# 鸢尾花
def Iris():
    iris = load_iris()

    iris_target = iris.target.reshape((150, 1))
    iris_data = np.hstack((iris.data, iris_target))
    iris_df = pd.DataFrame(iris_data)
    iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    print(iris_df.head())

    # 特征间相关性
    sns.pairplot(iris_df, hue='species', markers=['o', '^', 's'])
    plt.show()

    # 使用热力图可视化相关性
    iris_corr = iris_df.drop('species', axis=1).corr()
    sns.heatmap(iris_corr)
    plt.show()
