import os
import numpy as np
import cv2
from glob import glob
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import svm
import pickle

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
    for i in tqdm(range(len(points))):
        for j in range(len(points[i])):
            feature = points[i][j].reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    return im_features


def hist_visual(histogram, path, n_bins=10):
    plt.subplot(1,2,1)
    plt.imshow(plt.imread(path))
    plt.subplot(1,2,2)
    plt.hist(histogram, bins=n_bins)
    plt.show()



def read(labels, subdir='img', split=0.4):
    sift = cv2.SIFT_create()
    visual_vocab_imgs = []
    visual_dict_imgs = []
    Y = []
    paths = []
    for label in labels:
        # for path in glob(f'{os.path.realpath(".")}/img/test_img/*.png'):
        image_paths = sorted(glob(f'{os.path.realpath(".")}/{subdir}/{label}/*.png'))
        for ind, path in tqdm(enumerate(image_paths)):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, des = sift.detectAndCompute(image, None)
            if des is None:
                continue
            if ind < split * len(image_paths):
                visual_vocab_imgs.append(des)
            else:
                visual_dict_imgs.append(des)
                paths.append(path)
                Y.append(label)
    return visual_vocab_imgs, visual_dict_imgs, paths, Y


if __name__ == "__main__":
    # airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
    labels = [1, 2, 9, 7, 3]
    cluster_sizes = 100
    visual_vocab_imgs, visual_dict_imgs, paths, Y = read(labels)

    print("SIFT completed")
    print(f"Visual vocabulary images: {len(visual_vocab_imgs)}")
    print(f"Visual dictionary images: {len(visual_dict_imgs)}")

    descriptors = np.vstack(visual_vocab_imgs)
    kmeans = KMeans(n_clusters=cluster_sizes).fit(descriptors)
    print("cluster completed")
    print(kmeans)

    features = extractFeatures(kmeans, visual_dict_imgs, no_clusters=cluster_sizes)
    normalized = preprocessing.normalize(features)

    #hist_visual(normalized[0], paths[0])

    model = svm.SVC()
    model.fit(normalized, Y)
    pickle.dump(model, open('svm.pkl', 'wb'))

    _, visual_dict_imgs_test, paths_test, Y_test = read(labels, subdir='test', split=0.1)
    X_test = preprocessing.normalize(extractFeatures(kmeans, visual_dict_imgs_test, no_clusters=cluster_sizes))

    preds = model.predict(X_test)
    print(confusion_matrix(Y_test, preds))