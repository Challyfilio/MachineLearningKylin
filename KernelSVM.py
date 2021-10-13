from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def KernelSVM():
    X, y = make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #'''
    df_x = pd.DataFrame(X[:, 0], columns=['x'])
    df_y = pd.DataFrame(X[:, 1], columns=['y'])
    df_tag = pd.DataFrame(y, columns=['tag'])
    df = pd.concat([df_x, df_y, df_tag], axis=1)
    # print(df.head())

    sns.lmplot(x='x', y='y', hue='tag', data=df,
               height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
    plt.show()
    #'''

    kernel_list=['linear','rbf','poly','sigmoid']
    for i in kernel_list:
        model = SVC(kernel=i)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print(i+':'+str(accuracy_score(prediction, y_test)))
