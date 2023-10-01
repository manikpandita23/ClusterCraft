# KMeans Image Segmentation from Scratch

This project has implemented k-means clustering algorithm from using python.

## Table of Contents

- [What is KMeans clustering algorithm and how does it work](#what-is-kmeans-clustering-algorithm-and-how-does-it-work)
    - [It works by](#it-works-by)
- [Algorithm Implementation](#algorithm-implementation)
    - [Initialization](#initialization)
    - [Assignment](#assignment)
    - [Updating](#updating)
    - [Iteration and Stopping Condition](#iteration-and-stopping-condition)
<!-- - [Content of repository](#content-of-repository) -->
- [Dependence](#dependence)
- [Installation](#installation)
- [How to use](#how-to-use)
- [Contributing](#contributing)

## What is KMeans clustering algorithm and how does it work
K-Means clustering is a popular unsupervised machine learning algorithm used for data segmentation and clustering. It is primarily used to group data points into clusters based on their similarity or proximity.

Example:

Before:

![before_kmeans](<README resources/before_kmeans.png>)

After:

![after_kmeans](<README resources/after_kmeans.png>)

as we can see the nearby dots got grouped together

In image segmentation, it basically provides a segmented image, aka a less detailed version of the image. 

More clusters (value of k (in the project it can only lie between 5 and 15 (both included))) means more detail.

Example:

Before:

![before_kmeans_img_seg](<README resources/before_kmeans_img_seg.jpeg>)

After:

(using 2 clusters)

![after_kmeans_img_seg_2](<README resources/after_kmeans_img_seg_2.png>)

(using 5 clusters)

![after_kmeans_img_seg_5](<README resources/after_kmeans_img_seg_5.png>)



### It works by:

First staring with K initial cluster centroids (representative points). (Initialization step)

Then it assigns each data point to the nearest centroid, forming clusters. (Assignment step)

After which it recalculate the centroids by taking the mean of data points in each cluster. (Updating step)

Then it repeat last 2 steps (Assignment and Updating) until centroids no longer change significantly. (Iteration step and stopping condition)

Finally, we get K clusters with data points grouped by similarity.

## Algorithm Implementation
First we got the image and cluster number as input and then implement the algorithm by following process:

### Initialization
For selecting the initial 'k' cluster centroids, we take 'k' random pixels form the image's pixel grid.

<details>
<summary>Code-snippet</summary>
```python:
    def initialize_centroids(self, X):
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[random_indices]
        return centroids
```
</details>

### Assignment

Then we assign each pixel to its nearest cluster (the distance between a cluster and a pixel found by their **Euclidean Distance**)

<details>
<summary>Euclidean Distance</summary>
Euclidean Distance = √((x2 - x1)^2 + (y2 - y1)^2) 

here x2 and y2 are cluster's location and color distances respectively while x1 and x2 are the location and color of the pixel with which the distance is being found.
</details>

<details>
<summary>Code-snippet</summary>
```python:
    def assign_clusters(self, X):
        distances = pairwise_distances(X, self.centroids)
        labels = np.argmin(distances, axis=1)
        return labels
```
</details>

### Updating

Then we take the mean (by using the **mean formula**) of all the pixels in a cluster making the coordinates of the mean as the new centroid.

We do this for all 'k' clusters.

<details>
<summary>Mean Formula</summary>
Mean = (1 / n) * Σ(All Points in Cluster)

here mean new centroid, Σ(All Points in Cluster) means sum of all the points in the current cluster and n refers to the number of points in the cluster

here x2 and y2 are cluster's location and color distances respectively while x1 and x2 are the location and color of the pixel with which the distance is being found.
</details>

<details>
<summary>Code-snippet</summary>
```python:
    def update_centroids(self, X):
        new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
        return new_centroids
```
</details>

### Iteration and Stopping Condition

Now we repeat Assignment and Updating steps for 100 Iteration cycles.

<details>
<summary>Code-snippet</summary>
```python:
    for _ in range(self.max_iters):
        prev_centroids = self.centroids.copy()
        self.labels = self.assign_clusters(X)
        self.centroids = self.update_centroids(X)
        if np.linalg.norm(self.centroids - prev_centroids) < self.tol:
            break
```
</details>

<!-- ## Content of repository

|S. No|Name|Extension|Language|Contains|
|:---:|:--:|:-------:|:------:|:------:| -->


## Dependence

To run the project you will need the following:

1) [Python](https://www.python.org/downloads/)
2) Following python libraries (have pip installed)
    1) numpy (pip install numpy)
    2) matplotlib (pip install matplotlib)
    3) scikit-metrics (pip install scikit-metrics)
    4) flask (pip install Flask)

## Installation

To install just download the zip file extract it and done!

## How to use

1) Run [***app.py***](./app.py) file in your python compiler
2) Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
3) Select an image and cluster number 
4) Click on 'segment image' 
**AND DONE!** you will then get your KMeans segmented image

Remember higher the cluster number the more time it will take to provide you withe the final image.

## Contributing
See [Contributing](./CONTRIBUTING.md) for the contribution guidelines
