import os
import numpy as np
import cv2
from glob import glob
from sklearn.cluster import KMeans
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# cluster_sizes = [400, 1000, 4000]

# TODO: get multiple cluster sizes
# def kmeans_cluster(points, K):
#     descriptors = np.array(points[0])
#     for descriptor in points[1:]:
#         descriptors = np.vstack((descriptors, descriptor))
#
#     kmeans = [KMeans(n_clusters=k).fit(points) for k in K]
#     return kmeans
# # cluster_sizes = [400, 1000, 4000]


# Mostly copied this function from github!
def extractFeatures(kmeans, points, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(points[i])):
            feature = points[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features


def hist_visual(x, n_bins):
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    axs[1].hist(x, bins=n_bins, density=True)


if __name__ == "__main__":
    # airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
    labels = [1, 2, 9, 7, 3]
    cluster_sizes = 5
    load = 0
    descriptor_list = []
    no_images = 0
    if not load:
        sift = cv2.SIFT_create()
        descriptors = None
        Y = None
        for label in labels:
            # TODO: get all images after debugging
            # for path in tqdm(glob(f'{os.path.realpath(".")}/img/{label}/*.png')):
            for path in glob(f'{os.path.realpath(".")}/img/test_img/*.png'):
                no_images += 1
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                _, des = sift.detectAndCompute(image, None)
                if descriptors is None:
                    descriptors = des
                    Y = [label]*des.shape[0]

                descriptor_list.append(des)
                descriptors = np.vstack([descriptors, des])
                Y = np.hstack([Y, [label]*des.shape[0]])
        print("SIFT completed")
        np.savetxt('points.txt', descriptors)
    else:
        descriptors = np.loadtxt('points.txt')

    kmeans = KMeans(n_clusters=cluster_sizes).fit(descriptors)
    print("cluster completed")
    print(kmeans)

    label_count = len(labels)
    features = extractFeatures(kmeans, descriptor_list, no_images, no_clusters=cluster_sizes)
    normalized = preprocessing.normalize(features)
    # hist_visual(kmeans)
