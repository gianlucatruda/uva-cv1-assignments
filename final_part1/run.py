import os
import numpy as np
import cv2
from glob import glob
from datetime import datetime
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
def extractFeatures(kmeans, points, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for _ in range(len(points))])
    for i in range(len(points)):
        for j in range(len(points[i])):
            feature = points[i][j].reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    return im_features


def hist_visual(x, n_bins):
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    axs[1].hist(x, bins=n_bins, density=True)


if __name__ == "__main__":
    # airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
    labels = [1, 2, 9, 7, 3]
    cluster_sizes = 500
    sift = cv2.SIFT_create()
    visual_vocab_imgs = []
    visual_dict_imgs = []
    Y = []
    start = datetime.now()
    for label in labels:
        image_paths = glob(f'{os.path.realpath(".")}/img/{label}/*.png')
        for ind, path in tqdm(enumerate(image_paths)):
        #for path in glob(f'{os.path.realpath(".")}/img/test_img/*.png'):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, des = sift.detectAndCompute(image, None)
            if ind < 0.4 * len(image_paths):
                visual_vocab_imgs.append(des)
            else:
                visual_dict_imgs.append(des)
                Y.append(label)
        print(f"Visual vocabulary images of label {label}: {len(visual_vocab_imgs)}")
        print(f"Visual dictionary images of label {label}: {len(visual_dict_imgs)}")
    print("SIFT completed")
    print(f"Visual vocabulary images: {len(visual_vocab_imgs)}")
    print(f"Visual dictionary images: {len(visual_dict_imgs)}")

    descriptors = np.vstack(visual_vocab_imgs)
    kmeans = KMeans(n_clusters=cluster_sizes).fit(descriptors)
    print("cluster completed")
    print(kmeans)

    features = extractFeatures(kmeans, visual_dict_imgs, no_clusters=cluster_sizes)
    normalized = preprocessing.normalize(features)
    # hist_visual(kmeans)
