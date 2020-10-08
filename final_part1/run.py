import os
import numpy as np
import cv2
from glob import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import svm, datasets

cluster_sizes = [400, 1000, 4000]


def kmeans_cluster(points, K):
    descriptors = np.array(points[0])
    for descriptor in points[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    kmeans = [KMeans(n_clusters=k).fit(points) for k in K]
    return kmeans


def extractFeatures(kmeans, points, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(points[i])):
            feature = points[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

# def hist_visual(x, n_bins):
#     fig, axs = plt.subplots(1, 2, tight_layout=True)
#     axs[1].hist(x, bins=n_bins, density=True)


if __name__ == "__main__":
    # airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
    labels = [1, 2, 9, 7, 3]

    ## Read in the train files
    # dataset = [[cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in glob(f'{os.path.realpath(".")}/img/{label}/*.png')]
    #            for label in labels]
    dataset = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in glob(f'{os.path.realpath(".")}/img/test_img/*.png')]

    # cv2.imshow('window', dataset[0][0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("data read in")

    sift = cv2.SIFT_create()
    points = np.zeros((128))

    for label in labels:
        for path in glob(f'{os.path.realpath(".")}/img/{label}/*.png'):
        # for path in glob(f'{os.path.realpath(".")}/img/test_img/*.png'):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, des = sift.detectAndCompute(image, None)
            points = np.vstack([points, des])

    print("SIFT completed")

    kmeans = kmeans_cluster(points, cluster_sizes)
    print("cluster completed")
    print(kmeans)
    label_count = 7
    image_count = len(dataset)
    extractFeatures(kmeans, points, image_count, no_clusters=)
    # hist_visual(kmeans)
