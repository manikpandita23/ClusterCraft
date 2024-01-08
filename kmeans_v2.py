#Importing libraries
import numpy as np
from sklearn.metrics import pairwise_distances

class CustomKMeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X):
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[random_indices]
        return centroids
        
    def assign_clusters(self, X):
        distances = pairwise_distances(X, self.centroids)
        labels = np.argmin(distances, axis=1)
        return labels
        
    def update_centroids(self, X):
        new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
        return new_centroids
    
    # Fit function
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iters):
            prev_centroids = self.centroids.copy()
            
            self.labels = self.assign_clusters(X)
            self.centroids = self.update_centroids(X)
            
            if np.linalg.norm(self.centroids - prev_centroids) < self.tol:
                break

    # Prediction function          
    def predict(self, X):
        distances = pairwise_distances(X, self.centroids)
        labels = np.argmin(distances, axis=1)
        return labels
