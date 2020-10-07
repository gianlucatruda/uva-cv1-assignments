import os
import numpy as np
import cv2
from glob import glob
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# cluster_sizes = [400, 1000, 4000]


if __name__ == "__main__":
    # airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
    labels = [1, 2, 9, 7, 3]
    cluster_sizes = 5
    load = 0

    if not load:
        sift = cv2.SIFT_create()
        points = None
        for label in labels:
            for path in tqdm(glob(f'{os.path.realpath(".")}/img/{label}/*.png')):
            # for path in glob(f'{os.path.realpath(".")}/img/test_img/*.png'):
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                _, des = sift.detectAndCompute(image, None)
                if points is None:
                    points = des
                points = np.vstack([points, des])
        
        print("SIFT completed")
        np.savetxt('points.txt', points)
    else:
        points = np.loadtxt('points.txt')

    kmeans = KMeans(n_clusters=cluster_sizes).fit(points)
    print("cluster completed")
    print(kmeans)
