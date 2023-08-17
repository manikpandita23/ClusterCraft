import matplotlib
import matplotlib.image as img
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from kmeans import CustomKMeans

test_image = img.imread('test.jpg')
plt.imshow(test_image)
