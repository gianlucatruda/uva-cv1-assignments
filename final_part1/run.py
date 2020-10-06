import os
import numpy as np
import cv2
from glob import glob
from sklearn.cluster import KMeans
from sklearn import svm

# airplane: 1, bird: 2, ship: 9, horse: 7, car: 3
labels = [1, 2, 9, 7, 3]

## Read in the train files
dataset = [[cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in glob(f'{os.path.realpath(".")}/img/{label}/*.png')] for label in labels]

# cv2.imshow('window', dataset[0][0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sift = cv2.SIFT_create()
points = np.zeros((128))

for label in labels:
    for path in glob(f'{os.path.realpath(".")}/img/{label}/*.png'):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, des = sift.detectAndCompute(image, None)
        points = np.vstack([points, des])

points = points[1:]