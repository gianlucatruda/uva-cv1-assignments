import cv2
import random
import numpy as np
import scipy
import math

from keypoint_matching import match_keypoints

no_data_required = 3
no_params = 6


def ransac(coords, P=20, N=600, threshold=10):

    best_parameters = [0] * no_params
    best_inliers = -1

    for n in range(N):
        # print(f"{n} of {N} iterations...", end='\r')

        inliers = 0

        # Pick P matches at random from the total set of matches T
        sampling = random.choices(coords, k=P)

        # Construct A and b using the P pairs of points, then solve for x
        params = get_parameters(sampling)

        # Construct the transformation matrix M and vector t
        m1, m2, m3, m4, t1, t2 = params
        M = np.array([[m1, m2], [m3, m4]])
        t = np.array([t1, t2])

        # Apply the transformation using the parameters to all T points
        for (x, y), (xprime, yprime) in coords:
            xt, yt = M @ [x, y] + t
            # Check if inlier using Euclidean distance
            if np.sqrt((xt - xprime)**2 + (yt - yprime)**2) < threshold:
                inliers += 1

        # If this is a good fit, update parameters
        if inliers > best_inliers:
            best_inliers = inliers
            best_parameters = params

    return best_parameters


def get_parameters(data):
    A_components = []
    b_components = []

    # Get matrix A and vector b
    for (x, y), (xprime, yprime) in data:
        A_components.append(np.array([
            [x, y, 0, 0, 1, 0],
            [0, 0, x, y, 0, 1]
        ]))
        b_components.append(np.array([xprime, yprime]))

    A = np.vstack(A_components)
    b = np.hstack(b_components)

    # Get new param x = (m1, m2, m3, m4, t1, t2) by solving the equation Ax = b.
    x = np.linalg.pinv(A) @ b

    return x


# TODO: Get visualization to work
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

    print("Loading coordinates...", end='\r')
    coords, kp1, kp2 = match_keypoints(
        im1, im2,
        show_keypoints=False,
        show_matches=False,
        limit=None,
        ratio=0.2)

    best_params = ransac(
        coords,
        P=20,
        N=600,
        threshold=10)
