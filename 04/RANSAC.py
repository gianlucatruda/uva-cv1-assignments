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


def visualise(src, params, shape):

    # Unpack params
    m1, m2, m3, m4, t1, t2 = best_params
    M = np.array([[m1, m2], [m3, m4]])
    t = np.array([t1, t2])

    # Construct new image (y, x, 3)
    dst = np.zeros((shape[1], shape[0], 3))

    # Loop through pixels, transform, and write to new image
    for x in range(src.shape[1]):
        for y in range(src.shape[0]):
            # Apply transformation (x',y') = M.[x,y]T + shift
            xt, yt = M @ [x, y] + t
            xt, yt = round(xt), round(yt)
            if yt < dst.shape[0] and xt < dst.shape[1] and min(xt, yt) >= 0:
                dst[yt, xt] = src[y, x]
    cv2.imwrite('figs/transformed.png', dst)


if __name__ == '__main__':

    # Set shape of output images (width, height)
    DSTSHAPE = (950, 800)

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
    visualise(im1, best_params, DSTSHAPE)
    print("Image saved to figs/transformed.png\n")

    print("Visualising cv2.warpAffine...\n")
    # Consruct the transformation matrix the way CV2 likes it
    _M = np.hstack((M, t.reshape(2, 1)))
    dst = cv2.warpAffine(im1, _M, DSTSHAPE)
    cv2.imwrite('figs/warpaffine.png', dst)
    print("Image saved to figs/warpaffine.png\n")
