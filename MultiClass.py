import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def MultiClass():
    digits = load_digits()

    _, axes = plt.subplots(5, 10)

    axes = axes.reshape(1, -1)
    for ax, image in zip(axes[0], digits.images[:50]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r)

    plt.show()

    X_raw = digits.images
    Y = digits.target

    X = X_raw.reshape(X_raw.shape[0], -1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(penalty='l1', fit_intercept=True, solver='saga', tol=0.001, max_iter=1000)
    clf.fit(X_train, Y_train)

    score = clf.score(X_test, Y_test)
    print("Test score with L1 penalty:%.4f" % score)

    predicted = clf.predict(X_test)
    print("Classificatio report for classifier %s :\n %s\n"
          % (clf, metrics.classification_report(Y_test, predicted)))

    metrics.plot_confusion_matrix(clf, X_test, Y_test)
    plt.show()
