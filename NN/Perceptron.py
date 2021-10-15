import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import seaborn as sns


def Perceptron_1():
    data = pd.read_csv("NN/countries_data.csv")

    # 区分特征和标注
    X = data[['Services_of_GDP', 'Services_of_GDP']]
    Y = data['label']

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    clf = Perceptron(penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, shuffle=True)
    clf.fit(X_train, Y)

    prediction = clf.predict(X_train)
    data['prediction'] = prediction
    sns.scatterplot(data['Services_of_GDP'], data['ages65_of_total'],
                    hue=data['prediction'], style=data['label'],data=data)
    plt.show()
