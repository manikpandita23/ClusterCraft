# k-Means Image Segmentation from Scratch

This project has implemented k-means clustering algorithm for image segmentation from scratch.

## Table of Contents

- [What is k-means clustering algorithm and how does it work?](#what-is-kmeans-clustering-algorithm-and-how-does-it-work)
    - [It works by](#it-works-by)
- [Algorithm Implementation](#algorithm-implementation)
    - [Initialization](#initialization)
    - [Assignment](#assignment)
    - [Updating](#updating)
    - [Iteration and Stopping Condition](#iteration-and-stopping-condition)
<!-- - [Content of repository](#content-of-repository) -->
- [Dependence](#dependence)
- [Download](#download)
- [How to use](#how-to-use)
- [Contributing](#contributing)

## What is KMeans clustering algorithm and how does it work
K-Means clustering is a popular unsupervised machine learning algorithm used for data segmentation and clustering. It is primarily used to group data points into clusters based on their similarity or proximity.

### Example:

| Before K-Means                                          | After K-Means                                   |
| --------------------------------------------------------| ----------------------------------------------- |
| ![Before K-Means](README%20resources/before_kmeans.png) | ![After K-Means](README%20resources/after_kmeans.png) |



> Image segmentation with clustering is a computer vision technique that involves the use of clustering algorithms to partition an image into distinct regions or clusters based on similarities in pixel values. This process is particularly useful in tasks like object detection and scene analysis, where isolating and understanding different parts of an image is essential.
>
> **Higher *k* values lead to more detailed image, but at the expense of higher compute cost and runtime.**



### Example:

| Original Image                                            | 2 Clusters                                          | 5 Clusters                                          |
| --------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| ![Before K-Means](README%20resources/before_kmeans_img_seg.jpeg 'Before K-Means') | ![After K-Means (2 Clusters)](README%20resources/after_kmeans_img_seg_2.png 'After K-Means (2 Clusters)') | ![After K-Means (5 Clusters)](README%20resources/after_kmeans_img_seg_5.png 'After K-Means (5 Clusters)') |



### It works by:

- First starting with *k* initial cluster centroids (representative points). (Initialization step)
- Assign each data point to the nearest centroid, forming clusters. (Assignment step)
- After which it recalculates the centroid position by taking the mean of data points in each cluster. (Updating step)
- Repeat last 2 steps (Assignment and Updating) until centroids no longer change significantly. (Iteration step and stopping condition)
- Finally, we get *k* clusters with data points grouped by similarity.

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
  
$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$
    
Where:
- $(x_1$, $y_1)$ are coordinates of the first point.
- $(x_2$, $y_2)$ are coordinates of the second point.
- $d$ is the Euclidean distance between $(x_1$, $y_1)$ and $(x_2$, $y_2)$
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

We calculate the mean of distances all the pixels in a cluster to determine the new centroid position. This is done for all *k* clusters.

<details>
<summary>Mean Formula</summary>
 $Mean = \frac{1}{n} \sum d_{i}$
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

The assignment and updation steps are repeated until the centroids positional change is close to the tolerance limit `tol`. To avoid infinite loops, a limit of 100 iterations is set.

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


## Dependencies

Following are the project dependencies:

1) [Python](https://www.python.org/downloads/)
2) Following python libraries (have pip installed)
    1) numpy (``pip install numpy``)
    2) matplotlib (``pip install matplotlib``)
    3) scikit-learn (``pip install scikit-learn``)
    4) flask (``pip install Flask``)

## Download

Run ```git clone https://github.com/surtecha/KMeans-Segmentation-from-Scratch.git``` in your terminal and navigate the folder for `app.py`.

## How to use

1) Run [***app.py***](./app.py) in your preferred IDE or code editor.
2) Go to [http://127.0.0.1:5000](http://127.0.0.1:5000).
3) Choose an image from your directories and select a cluster count. 
4) Click on **Segment image** button to get the segmented image.

**Note:** Higher cluster count requires significantly higer computational power and time.

## Contributing
See [Contributing](./CONTRIBUTING.md) for the contribution guidelines.
