import cv2
import random
import numpy as np
import os

from keypoint_matching import match_keypoints

no_data_required = 3
no_params = 6


def ransac(coords, P=20, N=50, threshold=10):

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


def visualise(im1, params):

    # Unpack params
    m1, m2, m3, m4, t1, t2 = best_params
    M = np.array([[m1, m2], [m3, m4]])
    t = np.array([t1, t2])

    # Construct new image
    im = np.zeros((1000, 1200, 3))

    # Loop through pixels, transform, and write to new image
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            # Apply transformation (x',y') = M.[x,y]T + shift
            xt, yt = M @ [x, y] + t
            xt, yt = round(xt), round(yt)
            if yt < im1.shape[0] and xt < im1.shape[1]:
                im[y, x] = im1[yt, xt]
    cv2.imwrite('figs/transformed.png', im)


if __name__ == '__main__':
    if not os.path.exists('figs'):
        os.makedirs('figs')

    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')

    print("Loading coordinates...\n")
    coords, kp1, kp2 = match_keypoints(
        im1, im2,
        show_keypoints=False,
        show_matches=False,
        limit=None,
        ratio=0.2)

    print("Running RANSAC...\n")
    best_params = ransac(
        coords,
        P=15,
        N=50,
        threshold=5)

    print("Best params for matrix M and vector t:")
    m1, m2, m3, m4, t1, t2 = best_params
    M = np.array([[m1, m2], [m3, m4]])
    t = np.array([t1, t2])
    print(M)
    print(t)

    print("\nVisualising transformed image...\n")
    visualise(im2, best_params)
    print("Image saved to figs/transformed.png\n")
