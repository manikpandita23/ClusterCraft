import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from kmeans import CustomKMeans

image = mpl.image.imread('test.jpg')
plt.imshow(image)