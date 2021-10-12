import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def Regularized():
    train_size = 20
    test_size = 12
    train_X = np.random.uniform(low=0, high=1.2, size=train_size)
    test_X = np.random.uniform(low=0.1, high=1.3, size=test_size)
    train_y = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)
    test_y = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)
    plt.scatter(train_X, train_y, marker='s', label='train')
    plt.scatter(test_X, test_y, marker='o', label='test')
    plt.legend(loc=3)
    plt.show()

    poly = PolynomialFeatures(6)
    train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
    test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))
    model = Ridge(alpha=1.0)
    model.fit(train_poly_X, train_y)
    train_pred_y = model.predict(train_poly_X)
    test_pred_y = model.predict(test_poly_X)

    # plt.plot(test_poly_X, test_pred_y, label='train')
    # plt.show()
    print(mean_squared_error(train_pred_y, train_y))
    print(mean_squared_error(test_pred_y, test_y))
