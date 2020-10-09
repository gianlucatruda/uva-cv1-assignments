import os
import numpy as np
import cv2
from glob import glob
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
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
def extract_features(kmeans, points, no_clusters):
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


def binary(labels, i):
    v = np.zeros(len(labels))
    v[i] = 1
    return v


def read_and_prepare(labels, subdir='img', split=0.4):
    sift = cv2.SIFT_create()
    visual_vocab_imgs = []
    visual_dict_imgs = []
    Y = []
    paths = []
    for i in range(len(labels)):
        #image_paths = sorted(glob(f'{os.path.realpath(".")}/img/{10*labels[i]+labels[i]}/*.png'))
        image_paths = sorted(glob(f'{os.path.realpath(".")}/{subdir}/{labels[i]}/*.png'))
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
                Y.append(binary(labels,i))
    Y = np.vstack(Y)
    return visual_vocab_imgs, visual_dict_imgs, paths, Y



if __name__ == "__main__":
    # airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
    labels = [1, 2, 9, 7, 3]
    cluster_sizes = 400
    visual_vocab_imgs, visual_dict_imgs, paths, Y = read_and_prepare(labels)

    print("SIFT completed")
    print(f"Visual vocabulary images: {len(visual_vocab_imgs)}")
    print(f"Visual dictionary images: {len(visual_dict_imgs)}")

    descriptors = np.vstack(visual_vocab_imgs)
    kmeans = KMeans(n_clusters=cluster_sizes).fit(descriptors)
    print("cluster completed")
    print(kmeans)
    pickle.dump(kmeans, open('kmeans.pkl', 'wb'))


    features = extract_features(kmeans, visual_dict_imgs, no_clusters=cluster_sizes)
    normalized = preprocessing.normalize(features)

    hist_visual(normalized[0], paths[0])

    models = [svm.SVC() for _ in labels]
    for i in range(len(models)):
        models[i].fit(normalized, Y[:,i])
        pickle.dump(models[i], open(f'svm_{i}.pkl', 'wb'))

    _, visual_dict_imgs_test, paths_test, Y_test = read_and_prepare(labels, subdir='test', split=0)
    X_test = preprocessing.normalize(extract_features(kmeans, visual_dict_imgs_test, no_clusters=cluster_sizes))

    for i in range(len(models)):
        preds = models[i].predict(X_test)
        print(accuracy_score(Y_test[:, i], preds))
        print(confusion_matrix(Y_test[:, i], preds))