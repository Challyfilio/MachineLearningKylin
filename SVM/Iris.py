import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score

iris = load_iris()

iris_target = iris.target.reshape((150, 1))
iris_data = np.hstack((iris.data, iris_target))
iris_df = pd.DataFrame(iris_data)
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']


# print(iris_df.head())


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
    sub_data1 = iris_df[iris_df.species != 0]  # 0,2
    # print(sub_data1)
    sub_X1 = sub_data1.drop(['species', 'sepal_length', 'sepal_width'], axis=1)
    sub_Y1 = sub_data1.species

    clf1 = svm.LinearSVC()
    clf1.fit(sub_X1, sub_Y1)

    # 获取分类结果
    prediction = clf1.predict(sub_X1)
    # print(prediction)

    # 绘制分类超平面
    w = clf1.coef_
    b = clf1.intercept_
    x = np.linspace(2.5, 7.5, 100)  # 2.5,7.5/1.5,3.5
    y = (w[0, 0] * x + b[0]) / (-w[0, 1])

    plt.plot(x, y, 'r')

    c1 = sub_Y1 == 1  # 1,2/0,1
    c2 = sub_Y1 == 2
    colors = np.asarray([i for i in map(lambda a: 'yellowgreen' if a == 1 else 'steelblue', prediction)])

    plt.scatter(sub_X1.values[c1, 0], sub_X1.values[c1, 1], c=colors[c1], s=60, alpha=0.5, marker='s')
    plt.scatter(sub_X1.values[c2, 0], sub_X1.values[c2, 1], c=colors[c2], s=60, alpha=0.5, marker='o')

    plt.grid()
    plt.show()

    print('accuracy_score:' + str(accuracy_score(prediction, sub_Y1)))  # 评估


# 使用全部特征
def Iris_LSVC_All():
    X1 = iris_df.drop(['species'], axis=1)
    Y1 = iris_df.species

    clf2 = svm.LinearSVC()
    clf2.fit(X1, Y1)
    prediction = clf2.predict(X1)

    X1_tSNE = TSNE(n_components=2).fit_transform(X1)  # 降维数据

    '''
    c1 = Y1 == 0
    c2 = Y1 == 1
    c3 = Y1 == 2

    colors = ['steelblue', 'yellowgreen', 'purple']
    dye = lambda colors, x: np.asarray([colors[int(i)] for i in x])

    # plt.scatter(X1_tSNE[c1, 0], X1_tSNE[c1, 1], c=dye(colors, prediction[c1]), s=60, alpha=0.5, marker='s')
    # plt.scatter(X1_tSNE[c2, 0], X1_tSNE[c2, 1], c=dye(colors, prediction[c2]), s=60, alpha=0.5, marker='o')
    # plt.scatter(X1_tSNE[c3, 0], X1_tSNE[c3, 1], c=dye(colors, prediction[c3]), s=60, alpha=0.5, marker='^')
    '''

    df_α = pd.DataFrame(X1_tSNE[:, 0], columns=['x'])
    df_β = pd.DataFrame(X1_tSNE[:, 1], columns=['y'])
    df_γ = pd.DataFrame(prediction, columns=['prediction'])
    df = pd.concat([df_α, df_β, df_γ], axis=1)

    sns.scatterplot(df['x'], df['y'], hue=df['prediction'], style=Y1, markers=['s', 'o', '^'], data=df)

    plt.grid()
    plt.show()

    # 置信分
    confidence = clf2.decision_function(X1[:5])
    print(confidence)

    target_names = ['setosa', 'versicolor', 'virginica']
    print(classification_report(Y1, prediction, target_names=target_names, digits=5))


# 线性不可分
def Iris_SVC():
    X1 = iris_df.drop(['species'], axis=1)
    Y1 = iris_df.species

    clf3 = svm.SVC(kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf3.fit(X1, Y1)
    prediction = clf3.predict(X1)

    X1_tSNE = TSNE(n_components=2).fit_transform(X1)  # 降维数据

    '''
    c1 = Y1 == 0
    c2 = Y1 == 1
    c3 = Y1 == 2

    colors = ['steelblue', 'yellowgreen', 'purple']
    dye = lambda colors, x: np.asarray([colors[int(i)] for i in x])

    plt.scatter(X1_tSNE[c1, 0], X1_tSNE[c1, 1], c=dye(colors, prediction[c1]), s=60, alpha=0.5, marker='s')
    plt.scatter(X1_tSNE[c2, 0], X1_tSNE[c2, 1], c=dye(colors, prediction[c2]), s=60, alpha=0.5, marker='o')
    plt.scatter(X1_tSNE[c3, 0], X1_tSNE[c3, 1], c=dye(colors, prediction[c3]), s=60, alpha=0.5, marker='^')
    '''

    df_α = pd.DataFrame(X1_tSNE[:, 0], columns=['x'])
    df_β = pd.DataFrame(X1_tSNE[:, 1], columns=['y'])
    df_γ = pd.DataFrame(prediction, columns=['prediction'])
    df = pd.concat([df_α, df_β, df_γ], axis=1)

    sns.scatterplot(df['x'], df['y'], hue=df['prediction'], style=Y1, markers=['s', 'o', '^'], data=df)

    plt.grid()
    plt.show()

    target_names = ['setosa', 'versicolor', 'virginica']
    print(classification_report(Y1, prediction, target_names=target_names, digits=5))
