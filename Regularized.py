from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Regularized():
    alpha = []
    beta = []
    gamma = []
    train_size = 20
    test_size = 12
    train_X = np.random.uniform(low=0, high=1.2, size=train_size)
    test_X = np.random.uniform(low=0.1, high=1.3, size=test_size)
    train_y = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)
    test_y = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)
    plt.scatter(train_X, train_y, marker='s', label='train')
    plt.scatter(test_X, test_y, marker='o', label='test')
    for i in range(2, 7):
        alpha.append(i)
        poly = PolynomialFeatures(i)  # 生成i次多项式 2~6
        train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
        test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))
        model = Ridge(alpha=1)
        model.fit(train_poly_X, train_y)
        a = model.coef_
        b = model.intercept_
        # print(a)
        # print(b)
        x = np.linspace(0, 1.2, 500)
        y = 0
        for j in range(0, i + 1):
            # print('j=' + str(j))
            # print('a[j]=' + str(a[j]))
            y += a[j] * x ** j
        plt.plot(x, y + b[0], label='n=' + str(i))

        train_pred_y = model.predict(train_poly_X)
        test_pred_y = model.predict(test_poly_X)
        beta.append(mean_squared_error(train_pred_y, train_y))
        gamma.append(mean_squared_error(test_pred_y, test_y))

    plt.legend(loc=3)
    plt.show()

    df_α = pd.DataFrame(alpha, columns=['次数'])
    df_β = pd.DataFrame(beta, columns=['训练误差'])
    df_γ = pd.DataFrame(gamma, columns=['验证误差'])
    df = pd.concat([df_α, df_β, df_γ], axis=1)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('colheader_justify', 'left')
    print(df)
