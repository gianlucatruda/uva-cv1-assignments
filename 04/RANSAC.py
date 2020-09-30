from keypoint_matching import match_keypoints
import cv2
import random
import numpy as np

no_param = 6


def RANSAC(image_1, image_2, P=5, N=5):
    T = match_keypoints(image_1, image_2, show_matches=False)
    # print(f"Found {len(T)} matches")
    best_parameters = [0] * no_param
    for n in range(N):
        sampling = random.choices(T, k=P)
        dx = np.gradient(sampling[0])
        dy = np.gradient(sampling[1])
        vector_b = [dx, dy]
        matrix_A = [[sampling[0], sampling[1], 0, 0, 1, 0], [0, 0, sampling[0], sampling[1], 0, 1]]
        # (m1, m2, m3, m4, t1, t2) by solving the equation Ax = b.
        new_param = np.linalg.inv(matrix_A) * vector_b
        print(new_param)


if __name__ == '__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')
    RANSAC(im1, im2)

