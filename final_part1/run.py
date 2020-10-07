import os
import numpy as np
import cv2
from glob import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# cluster_sizes = [400, 1000, 4000]


def kmeans_cluster(points, K):
    descriptors = np.array(points[0])
    for descriptor in points[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # kmeans = [KMeans(n_clusters=k).fit(points) for k in K]
    kmeans = KMeans(n_clusters=2).fit(descriptors)
    return kmeans


if __name__ == "__main__":
    # airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
    labels = [1, 2, 9, 7, 3]

    sift = cv2.SIFT_create()
    points = np.zeros((128))

    for label in labels:
        # for path in glob(f'{os.path.realpath(".")}/img/{label}/*.png'):
        for path in glob(f'{os.path.realpath(".")}/img/test_img/*.png'):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, des = sift.detectAndCompute(image, None)
            points = np.vstack([points, des])

    print("SIFT completed")

    kmeans = kmeans_cluster(points, cluster_sizes)
    print("cluster completed")
    print(kmeans)
