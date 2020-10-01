from keypoint_matching import match_keypoints
import cv2
import random
import numpy as np
import scipy
import math

no_param = 6
no_data_required = 3


def RANSAC(image_1, image_2, P=20, N=100, RANSAC_TRESHOLD=200):
    coords, kp1, kp2 = match_keypoints(image_1, image_2,
                                    show_keypoints=True,
                                    limit=10,
                                    ratio=0.2)

    best_parameters = [0] * no_param
    best_fit = None
    inliers = []

    for n in range(N):
        sampling = random.choices(coords, k=P)
        # Is maybeinliers
        new_param = get_parameters(sampling)
        # Is maybemodel

        also_inliers = []
        best_error = 500

        # Get other inlier points
        for (x, y), (xprime, yprime) in coords:
            if ((x, y), (xprime, yprime)) not in sampling:
                # sum squared error
                point_error = math.sqrt((x-xprime)**2)+((y-yprime)**2)
                if point_error < RANSAC_TRESHOLD:
                    also_inliers.append([(x, y), (xprime, yprime)])

        # Check if model is good. If so, check how good.
        if len(also_inliers) > no_data_required:
            new_model = get_parameters(sampling + also_inliers)

            new_error = 0
            if new_error < best_error:
                best_fit = new_model
                best_parameters = new_param
                inliers = sampling + also_inliers

        print(best_parameters)

    bf = cv2.BFMatcher()
    inliers_t = np.transpose(inliers)
    matches = bf.match(inliers_t[0], inliers_t[1])

    visualize(kp1, kp2, matches)

    return best_parameters, best_fit


def fit(data, input, output):
    A = np.vstack([data[:, i] for i in input]).T
    B = np.vstack([data[:, i] for i in output]).T
    x, resids, rank, s = scipy.linalg.lstsq(A, B)
    return x

# def get_error(data, model):
#     A = np.vstack([data[:,i] for i in input_columns]).T
#     B = np.vstack([data[:,i] for i in output_columns]).T
#     B_fit = np.dot(A, model)
#     err_per_point = np.sum((B-B_fit)**2,axis=1) # sum squared error per row
#     return err_per_point


def get_parameters(data):
    A_components = []
    b_components = []

    # Get matrix A and vector b
    for (x, y), (xprime, yprime) in data:
        A_components.append(np.array([
            [x, y, 1, 0, 0, 0],
            [0, 0, 0, x, y, 1]
        ]))
        b_components.append(np.array([xprime, yprime]))

    A = np.vstack(A_components)
    b = np.hstack(b_components)

    # Get new param x = (m1, m2, m3, m4, t1, t2) by solving the equation Ax = b.
    new_param = np.linalg.pinv(A) * b
    # = maybemodel

    return new_param


def visualize(kp1, kp2, good):
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(
        im1, kp1, im2, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matching', img3)
    cv2.imwrite('figs/matches.png', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img1 = cv2.drawKeypoints(
        im1, kp1, im1,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(
        im2, kp2, im2,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image1', img1)
    cv2.imshow('image2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')
    best_param, best_fit = RANSAC(im1, im2)

