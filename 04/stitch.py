import numpy as np
import cv2

from keypoint_matching import match_keypoints
from sklearn.linear_model import RANSACRegressor # TODO switch to ours


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

    import ipdb; ipdb.set_trace()



    # 2. estimate the size of the stitched image by calculating the transformed
    # coordinates of the second image

    new_x, new_y = im2_trans.shape

    img = np.zeros((new_x, new_y))

    # 3. combine the left.jpg with the transformed right.jpg in one image

    return img


if __name__ == "__main__":
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')

    stitch_images(im1, im2)
