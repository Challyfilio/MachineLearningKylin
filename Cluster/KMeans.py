from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans

nb_samples = 1000


def show_dataset_1(X):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[:, 0], X[:, 1], marker='o')
    plt.show()


def show_dataset_2(X, Y):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(nb_samples):
        if Y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')
    plt.show()


def show_clustered_dataset_1(X, km):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(nb_samples):
        c = km.predict(X[i].reshape(1, -1))
        if c == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        elif c == 1:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='s', color='g')
    plt.show()


def show_clustered_dataset_2(X, km):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    for i in range(nb_samples):
        c = km.predict(X[i].reshape(1, -1))
        if c == 0:
            ax.scatter(X[i, 0], X[i, 1], marker='o', color='r')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='^', color='b')

    plt.show()


def KMeans_1():
    X, _ = make_blobs(n_samples=nb_samples, n_features=2, centers=3,
                      cluster_std=1.5, random_state=1000)
    show_dataset_1(X)

    km = KMeans(n_clusters=3)
    km.fit(X)

    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=3, n_init=10,
           random_state=None, tol=0.0001, verbose=0)
    # n_jobs=1, precompute_distances='auto',
    print(km.cluster_centers_)

    show_clustered_dataset_1(X, km)


def KMeans_2():
    X, Y = make_circles(n_samples=nb_samples, noise=0.05)
    show_dataset_2(X, Y)

    km = KMeans(n_clusters=2)
    km.fit(X)

    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
           n_clusters=2, n_init=10,
           random_state=None, tol=0.0001, verbose=0)
    print(km.cluster_centers_)

    show_clustered_dataset_2(X, km)
