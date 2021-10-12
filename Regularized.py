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
    for i in range(2, 7):
        print(i)
        poly = PolynomialFeatures(i)
        train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
        test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))
        model = Ridge(alpha=1)
        model.fit(train_poly_X, train_y)
        a = model.coef_
        b = model.intercept_
        print(a)
        print(b)
        x = np.linspace(0, 1.2, 100)
        for j in range(0, i + 1):
            print(j)
            y = a[j] * x ** j
        plt.plot(x, y + b, label='n=' + str(i))


        
    plt.legend(loc=3)
    plt.show()


'''
    poly = PolynomialFeatures(4)
    train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
    test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))

    model = Ridge(alpha=1.0)
    model.fit(train_poly_X, train_y)
    b = model.intercept_
    a = model.coef_
    print(a)
    print(b)

    x = np.linspace(0, 1.2, 100)
    plt.plot(x, a[2] * x ** 2 + a[1] * x + b, label='n=2')
    plt.plot(x, a[3] * x ** 3 + a[2] * x ** 2 + a[1] * x + b, label='n=3')
    plt.plot(x, a[4] * x ** 4 + a[3] * x ** 3 + a[2] * x ** 2 + a[1] * x + b, label='n=3')

    train_pred_y = model.predict(train_poly_X)
    test_pred_y = model.predict(test_poly_X)

    plt.scatter(train_X, train_y, marker='s', label='train')
    plt.scatter(test_X, test_y, marker='o', label='test')
    plt.legend(loc=3)

    # x= np.linspace(0,1.2,20)
    # plt.plot(train_X, train_pred_y, label='train')
    plt.show()
    print(mean_squared_error(train_pred_y, train_y))
    print(mean_squared_error(test_pred_y, test_y))
'''
