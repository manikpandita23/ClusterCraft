import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from sklearn.cluster import KMeans as km

from kmeans_v2 import CustomKMeans as ckm

def list_image_files(folder_path):
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    image_files = [file for file in os.listdir(folder_path) if os.path.splitext(file)[1].lower() in image_extensions]
    return image_files

def segment_image(selected_image_path):
    image = plt.imread(selected_image_path)

    plt.figure(figsize=(15, 5))
    plt.title('Original Image')
    plt.imshow(image)
    plt.show()

    num_clusters = int(input("Clusters: "))

    # Scikit-learn segmentation
    X = image.reshape(-1, 3)

    scikit_start = time.time()
    kmeans_scikit = km(n_clusters=num_clusters, n_init=10)
    kmeans_scikit.fit(X)
    segmented_img_1 = kmeans_scikit.cluster_centers_[kmeans_scikit.labels_]
    segmented_img_1 = segmented_img_1.reshape(image.shape)
    scikit_end = time.time()

    plt.figure(figsize=(15, 5))
    plt.title('Segmented Image (Scikit-learn)')
    plt.imshow(segmented_img_1)
    plt.show()    

    # Custom segmentation
    n_pixels = image.shape[0] * image.shape[1]
    image_pixels = image.reshape(n_pixels, -1)

    custom_start = time.time()
    kmeans_custom = ckm(n_clusters=num_clusters)
    kmeans_custom.fit(image_pixels)
    cluster_labels = kmeans_custom.predict(image_pixels)
    segmented_img_2 = kmeans_custom.centroids[cluster_labels].reshape(image.shape)
    custom_end = time.time()

    plt.figure(figsize=(15, 5))
    plt.title('Segmented Image (Custom)')
    plt.imshow(segmented_img_2)
    plt.show()

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    # Segmentation using scikit-learn KMeans
    plt.subplot(132)
    plt.title("Segmentation (scikit-learn KMeans)")
    plt.imshow(segmented_img_1.astype(np.uint8))
    plt.axis('off')

    # Segmentation using custom KMeans
    plt.subplot(133)
    plt.title("Segmentation (custom KMeans)")
    plt.imshow(segmented_img_2.astype(np.uint8))
    plt.axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    print(f"Scikit-learn KMeans took {scikit_end - scikit_start} seconds.")
    print(f"Custom KMeans took {custom_end - custom_start} seconds.")

images_folder = '/home/aetherlock/Pictures'
image_files = list_image_files(images_folder)

print("Available image files in the Downloads folder:")
for idx, file in enumerate(image_files, start=1):
    print(f"{idx}. {file}")

selected_index = int(input("Enter the index of the image to perform segmentation on: ")) - 1

if 0 <= selected_index < len(image_files):
    selected_image_path = os.path.join(images_folder, image_files[selected_index])
    print(f"Selected image: {selected_image_path}")
    segment_image(selected_image_path)
else:
    print("Invalid index.")
