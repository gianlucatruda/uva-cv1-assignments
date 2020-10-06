import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn import svm


## Read in the train files
with open('stl10_binary/train_X.bin', 'rb') as f:
    data = f.read()
    # CV2
    data = np.fromstring(data, np.uint8).reshape(-1, 96, 96, 3)
    sift = cv2.SIFT_create()
    points = np.zeros((128))
    for image in data:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        points = np.vstack([points, des])
        #cv2.imshow('window', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

