from keypoint_matching import match_keypoints
import cv2
import random
import numpy as np

no_param = 6


def RANSAC(image_1, image_2, P=10, N=10, RANSAC_TRESHOLD=3):
    coords, kp1, kp2 = match_keypoints(image_1, image_2,
                                    show_keypoints=True,
                                    limit=10,
                                    ratio=0.2)

    best_parameters = [0] * no_param
    highest_no = 0
    for n in range(N):
        sampling = random.choices(coords, k=P)
        A_components = []
        b_components = []

        for (x, y), (xprime, yprime) in sampling:
            A_components.append(np.array([
                [x, y, 1, 0, 0, 0],
                [0, 0, 0, x, y, 1]
            ]))
            b_components.append(np.array([xprime, yprime]))

        A = np.vstack(A_components)
        b = np.hstack(b_components)

        # Get new param = (m1, m2, m3, m4, t1, t2) by solving the equation Ax = b.
        new_param = np.linalg.pinv(A) * b

        inliers = []
        x_list = []
        y_list = []
        num = 0

        # find lines to the model for all testing points
        for i in range(sampling.shape[0]):
            #Get some shit done here
            



        x_inliers = np.array(x_list)
        y_inliers = np.array(y_list)

        new_no = len(inliers)
        if new_no > highest_no:
            highest_no = new_no
            best_parameters = new_param
        print(best_parameters)


# def fit(A, Y):
#     A_transpose = A.transpose()
#     ATA = A_transpose.dot(A)
#     ATY = A_transpose.dot(Y)
#     model = (np.linalg.inv(ATA)).dot(
#         ATY)  ## For a linear eq. AP = Y to solve a least sqaure problem,  P = (inverse(A'A))(A'Y)
#     return model


if __name__ == '__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')
    RANSAC(im1, im2)

