from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()

# 区分特征和标注
X_raw = digits.images
Y = digits.target
# 三维矩阵转为二维
X = X_raw.reshape(X_raw.shape[0], -1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# 归一化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#identity：none、logistic：sigmoid、tanh、relu
clf = MLPClassifier(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', max_iter=1000, alpha=0.7)
clf.fit(X_train, Y_train)
