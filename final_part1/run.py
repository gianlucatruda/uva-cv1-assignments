import os
import numpy as np
import cv2
from glob import glob
from sklearn.cluster import KMeans
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
import pandas as pd

from eval import calculate_mAP


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


def read_and_prepare(labels, desc_type, subdir='img', split=0.4):
    if desc_type in ['SIFT_RGB', 'SIFT_GRAY']:
        sift = cv2.xfeatures2d.SIFT_create()
    elif desc_type == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)
    else:
        raise NotImplemented("No such descriptor type")
    visual_vocab_imgs = []
    visual_dict_imgs = []
    Y = []
    paths = []
    for i in range(len(labels)):
        #image_paths = sorted(glob(f'{os.path.realpath(".")}/img/{10*labels[i]+labels[i]}/*.png'))
        image_paths = list(sorted(glob(f'{os.path.realpath(".")}/{subdir}/{labels[i]}/*.png')))
        for ind, path in tqdm(enumerate(image_paths)):
            if desc_type == 'SIFT_RGB':
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                # Concatenation doesn't work since des1, des2 and des3 have different sizes
                # _, des1 = sift.detectAndCompute(image[:,:,0], None)
                # _, des2 = sift.detectAndCompute(image[:,:,1], None)
                # _, des3 = sift.detectAndCompute(image[:,:,2], None)
                # des = np.hstack([des1, des2, des3])
                _, des = sift.detectAndCompute(image, None)
            elif desc_type == 'SIFT_GRAY':
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                _, des = sift.detectAndCompute(image, None)
            elif desc_type == 'SURF':
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                _, des = surf.detectAndCompute(image, None)
            if des is None:
                continue
            if ind < split * len(image_paths):
                visual_vocab_imgs.append(des)
            else:
                visual_dict_imgs.append(des)
                paths.append(path)
                Y.append(binary(labels, i))
    Y = np.vstack(Y)
    return visual_vocab_imgs, visual_dict_imgs, paths, Y



def run(cluster_size = 400, desc_type='SIFT_GRAY'):
    '''
    cluster_size: size of the visual vocabulary
    desc_type: type of descriptor, can be 'SIFT_GRAY', 'SIFT_RGB', 'SURF'
    '''

    print(f"Running {desc_type} with {cluster_size}-sized clusters...\n\n")

    # airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
    labels = [1, 2, 9, 7, 3]
    visual_vocab_imgs, visual_dict_imgs, paths, Y = read_and_prepare(labels, desc_type)

    print(f"{desc_type} completed")
    print(f"Visual vocabulary images: {len(visual_vocab_imgs)}")
    print(f"Visual dictionary images: {len(visual_dict_imgs)}")

    print("Clustering with K-means...")
    descriptors = np.vstack(visual_vocab_imgs)
    kmeans = KMeans(n_clusters=cluster_size).fit(descriptors)
    print("K-means completed")
    pickle.dump(kmeans, open(f'{cluster_size}-{desc_type}-kmeans.pkl', 'wb'))
    print('Saved clusters to pickle file')

    print('Preprocessing...')
    features = extract_features(kmeans, visual_dict_imgs, no_clusters=cluster_size)
    normalized = preprocessing.normalize(features)

    hist_visual(normalized[0], paths[0])

    print('Training binary models')
    models = [svm.SVC() for _ in labels]
    for i in tqdm(range(len(models))):
        models[i].fit(normalized, Y[:,i])
        pickle.dump(models[i], open(f'{cluster_size}-{desc_type}-svm_{i}.pkl', 'wb'))

    # prepare files for evaluation
    print('Preparing test data')
    _, visual_dict_imgs_test, paths_test, Y_test = read_and_prepare(labels, desc_type, subdir='test', split=0)
    X_test = preprocessing.normalize(extract_features(kmeans, visual_dict_imgs_test, no_clusters=cluster_size))
    np.savetxt(f'{cluster_size}-{desc_type}-X_test.txt', X_test)
    np.savetxt(f'{cluster_size}-{desc_type}-Y_test.txt', Y_test)
    np.savetxt(f'{cluster_size}-{desc_type}-paths_test.txt', paths_test, fmt="%s")
    print("Test data saved to txt files")
    dfs = []
    for i in range(len(models)):
        preds = models[i].predict(X_test)
        p = np.array(models[i].decision_function(X_test))
        df = pd.DataFrame(list(zip(preds, Y_test[:, i], preds == Y_test[:, i], p, paths_test)),
                          columns=['preds', 'truth', 'correct', 'votes', 'paths'])
        dfs.append(df)
    print(f"mAP for this setting: {calculate_mAP(dfs)}")


if __name__ == "__main__":
    print('Experiment 1: Cluster sizes')
    for size in [400, 1000, 4000]:
        run(cluster_size=size)
    run(400, desc_type='SIFT_RGB')
    run(400, desc_type='SURF')