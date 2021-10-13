from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def KernelSVM():
    X,y=make_gaussian_quantiles(n_features=2,n_classes=2,n_samples=100)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    model=SVC()
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    print(accuracy_score(prediction,y_test))