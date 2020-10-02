import numpy as np
import cv2
import os

from keypoint_matching import match_keypoints
from RANSAC import ransac


def stitch_images(im1, im2):

    # 1. find the best transformation between input images (keypoint_matching + RANSAC)
    coords, kp1, kp2 = match_keypoints(im1, im2, show_matches=False, ratio=0.2)

    # Build A and b from matches
    A_components = []
    b_components = []
    for (x, y), (xprime, yprime) in coords:
        A_components.append(np.array([
            [x, y, 0, 0, 1, 0],
            [0, 0, x, y, 0, 1]
        ]))
        b_components.append(np.array([xprime, yprime]))

    A = np.vstack(A_components)
    b = np.hstack(b_components)

    # Solve for t using RANSAC
    t = ransac(coords, P=20, N=100, threshold=3)

    # Build transformation matrix and shift vector
    m1, m2, m3, m4, t1, t2 = t
    M = np.array([[m1, m2], [m3, m4]])
    shift = np.array([t1, t2])

    print("Transformation matrix:", M, sep='\n')
    print("Translation vector:", shift, sep='\n')

    # 2. estimate the size of the stitched image by calculating the transformed
    # coordinates of the second image
    ymax, xmax, _ = im2.shape
    new_corners = []

    # Loop through the old corners of im2
    for xt, yt in np.array([[0, 0], [xmax, 0],
                            [0, ymax], [xmax, ymax]]):
        # Subtract the shift and then use inverse of M
        x, y = np.linalg.inv(M) @ ([xt, yt] - shift.T)
        new_corners.append([x, y])
    new_corners = np.array(new_corners)

    # Take extreme values
    xmax = round(new_corners[:, 0].max()) + 1
    ymax = round(new_corners[:, 1].max()) + 1
    new_x, new_y = max(im1.shape[1], xmax), max(im1.shape[0], ymax)

    # Construct new image of these dimensions
    img = np.zeros((new_y, new_x, 3))

    # 3. combine the left.jpg with the transformed right.jpg in one image
    for x in range(new_x):
        for y in range(new_y):
            # Apply transformation (x',y') = M.[x,y]T + shift
            xt, yt = M @ [x, y] + shift.T
            xt, yt = round(xt), round(yt)

            # Fill in pixels from right image
            if xt < im2.shape[1] and yt < im2.shape[0] and min(xt, yt) >= 0:
                img[y, x] = im2[yt, xt]
            # Fill in pixels from left image
            if x < im1.shape[1] and y < im1.shape[0]:
                img[y, x] = im1[y, x]

    return img


if __name__ == "__main__":

    if not os.path.exists('figs'):
        os.makedirs('figs')

    im1 = cv2.imread('left.jpg')
    im2 = cv2.imread('right.jpg')

    im = stitch_images(im1, im2)
    cv2.imwrite('figs/stitched.png', im)
    print('Saved results in figs/stitched.png')
