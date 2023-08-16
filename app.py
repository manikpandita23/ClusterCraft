from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import CustomKMeans

centroids = [(-5, 5), (5, 5)]
clusters_std = [1, 1]

X, y = make_blobs(n_samples=100, 
                  cluster_std=clusters_std, 
                  centers=centroids, 
                  n_features=2, 
                  random_state=2)

km = CustomKMeans(n_clusters=2, max_iter=100)
km.fit_predict(X)