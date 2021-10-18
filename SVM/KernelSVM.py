from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def KernelSVM():
    X, y = make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    '''
    df_x = pd.DataFrame(X[:, 0], columns=['x'])
    df_y = pd.DataFrame(X[:, 1], columns=['y'])
    df_tag = pd.DataFrame(y, columns=['tag'])
    df = pd.concat([df_x, df_y, df_tag], axis=1)
    # print(df.head())

    sns.lmplot(x='x', y='y', hue='tag', data=df,
               height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
    plt.show()
    '''

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    w = model.coef_
    b = model.intercept_
    print(w)
    print(b)
    prediction = model.predict(X_test)

    x = np.linspace(-2, 2, 100)
    yl = (w[0, 0] * x + b[0]) / (-w[0, 1])

    '''
    c1 = y_test == 0
    c2 = y_test == 1
    colors = np.asarray([i for i in map(lambda a: 'yellowgreen' if a == 1 else 'steelblue', prediction)])

    plt.scatter(X_test[c1, 0], X_test[c1, 1], c=colors[c1], s=60, alpha=0.5, marker='s')
    plt.scatter(X_test[c2, 0], X_test[c2, 1], c=colors[c2], s=60, alpha=0.5, marker='o')
    '''

    df_α = pd.DataFrame(X_test[:, 0], columns=['x'])
    df_β = pd.DataFrame(X_test[:, 1], columns=['y'])
    df_γ = pd.DataFrame(prediction, columns=['prediction'])
    df_δ = pd.DataFrame(y_test, columns=['species'])
    df = pd.concat([df_α, df_β, df_γ, df_δ], axis=1)

    sns.scatterplot(x=df['x'], y=df['y'], hue=df['prediction'], style=df['species'], markers=['s', 'o'], data=df)
    plt.title('linear')
    plt.plot(x, yl, 'r')
    plt.show()

    print('linear:' + str(accuracy_score(prediction, y_test)))

    kernel_list = ['rbf', 'poly', 'sigmoid']
    for i in kernel_list:
        model = SVC(kernel=i)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        df_α = pd.DataFrame(X_test[:, 0], columns=['x'])
        df_β = pd.DataFrame(X_test[:, 1], columns=['y'])
        df_γ = pd.DataFrame(prediction, columns=['prediction'])
        df_δ = pd.DataFrame(y_test, columns=['species'])
        df = pd.concat([df_α, df_β, df_γ, df_δ], axis=1)

        sns.scatterplot(x=df['x'], y=df['y'], hue=df['prediction'], style=df['species'], markers=['s', 'o'], data=df)
        plt.title(i)
        plt.show()

        print(i + ':' + str(accuracy_score(prediction, y_test)))