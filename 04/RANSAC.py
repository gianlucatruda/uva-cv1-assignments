from keypoint_matching import match_keypoints
import cv2
import random
import numpy as np

no_param = 6


def RANSAC(image_1, image_2, P=5, N=5):
    matches, kp1, kp2 = match_keypoints(image_1, image_2, show_matches=False)
    matches_coord = []
    for match in matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        matches_coord.append([p1, p2])

    print(matches_coord)
    T = matches
    best_parameters = [0] * no_param
    highest_no = 0
    for n in range(N):
        sampling = random.choices(T, k=P)
        # dx = transform(sampling[0])
        # dy = transform(sampling[1])
        # vector_b = [dx, dy]
        matrix_A = [[sampling[0], sampling[1], 0, 0, 1, 0], [0, 0, sampling[0], sampling[1], 0, 1]]
        # (m1, m2, m3, m4, t1, t2) by solving the equation Ax = b.
        new_param = np.linalg.inv(matrix_A) * vector_b

        inliers = []

        new_no = len(inliers)
        if new_no > highest_no:
            highest_no = new_no
            best_parameters = new_param
        print(best_parameters)


if __name__ == '__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')
    RANSAC(im1, im2)

