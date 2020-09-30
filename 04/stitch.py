import numpy as np
import cv2
import os

from keypoint_matching import match_keypoints
from sklearn.linear_model import RANSACRegressor  # TODO switch to ours


def stitch_images(im1, im2):

    # 1. find the best transformation between input images (keypoint_matching + RANSAC)
    coords, kp1, kp2 = match_keypoints(im1, im2, show_matches=False)

    # TODO Build A and b from matches
    A_components = []
    b_components = []
    for (x, y), (xprime, yprime) in coords:
        A_components.append(np.array([
            [x, y, 1, 0, 0, 0],
            [0, 0, 0, x, y, 1]
        ]))
        b_components.append(np.array([xprime, yprime]))

    A = np.vstack(A_components)
    b = np.hstack(b_components)

    # Solve for t using RANSAC (TODO switch to our own later)
    t = RANSACRegressor().fit(A, b).estimator_.coef_

    # Build transformation matrix and shift vector
    m1, m2, t1, m3, m4, t2 = t
    M = np.array([[m1, m2], [m3, m4]])
    shift = np.array([t1, t2])


    # 2. estimate the size of the stitched image by calculating the transformed
    # coordinates of the second image

    xmax, ymax, _ = im2.shape
    new_corners = []
    for p in np.array([[0, 0], [xmax, 0],
                       [0, ymax], [xmax, ymax]]):
        p_trans = M @ p.T + shift.T
        new_corners.append(p_trans)
    new_corners = np.array(new_corners)


    xmin, xmax = new_corners[:, 0].min(), new_corners[:, 0].max()
    ymin, ymax = new_corners[:, 1].min(), new_corners[:, 1].max()


    # new_x, new_y = round(im1.shape[0] + xmax - xmin) + 1, round(im1.shape[1] + ymax - ymin) + 1
    new_x, new_y = 1000, 1000
    img = np.zeros((new_x, new_y, 3))

    # 3. combine the left.jpg with the transformed right.jpg in one image
    for x in range(new_x):
        for y in range(new_y):
            if x < im1.shape[0] and y < im1.shape[1]:
                # img[x, y] = im1[x, y]
                pass

            xt, yt = M @ [x, y] + shift.T
            xt, yt = round(xt), round(yt)

            if xt < im2.shape[0] and yt < im2.shape[1]:
                img[x, y] = im2[xt, yt]


    return img


if __name__ == "__main__":

    if not os.path.exists('figs'):
        os.makedirs('figs')


    im1 = cv2.imread('left.jpg')
    im2 = cv2.imread('right.jpg')

    im = stitch_images(im1, im2)
    cv2.imwrite('figs/stitched.png', im)
