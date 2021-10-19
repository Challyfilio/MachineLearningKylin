from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

def show_dataset(X):
    fig,ax=plt.subplots(1,1,figsize=(15,10))
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[:,0],X[:,1],marker='o')
    plt.show()

def KMeans():
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=3,
                      cluster_std=1.5, random_state=1000)
    print(X)
    show_dataset(X)