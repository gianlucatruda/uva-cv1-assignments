# For this assignment, you will be given two pairs of images:
# synth1.pgm, synth2.pgm; and sphere1.ppm, sphere2.ppm.
# You should estimate the optical flow between these two pairs.
# That is, you will get optical flow for sphere images, and for synth images separately.
# Implement the Lucas-Kanade algorithm using the following steps.
# 1. Divide input images on non-overlapping regions, each region being 15 × 15.
# 2. For each region compute A, AT and b. Then, estimate optical flow as given in Equation 22.
# 3. When you have estimation for optical flow (Vx, Vy) of each region, you should display the results.
# There is a matplotlib function quiver which plots a set of two-dimensional vectors as arrows on the screen.
# Try to figure out how to use this to show your optical flow results.
# Note: You are allowed to use scipy.signal.convolve2d to perform convolution.
# Include a demo function to run your code

from cv2 import imread
import numpy as np
from scipy import signal
from skimage.util.shape import view_as_windows
from matplotlib import pyplot as plt

block_shape = (15, 15)


def get_nonoverlapping_regions(image_1, image_2):

    image_1 = np.mean(image_1, axis=2)
    image_2 = np.mean(image_2, axis=2)

    blocks_1 = view_as_windows(image_1.astype(np.uint8), block_shape, step=(15, 15))
    blocks_2 = view_as_windows(image_2.astype(np.uint8), block_shape, step=(15, 15))
    print(blocks_1.shape)
    wind = (int(image_1.shape[0]/block_shape[0]), int(image_1.shape[1]/block_shape[1]))

    # Loop over regions and get A A^t and b for every region.
    for first in range(wind[0]):
        for second in range(wind[1]):
            # plt.imshow(blocks_1[first][second])
            # plt.show()
            get_block_A_and_b(blocks_1[first][second], blocks_2[first][second])


# TODO: very long function, maybe split up
def get_block_A_and_b(img1, img2):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])

    fx = signal.convolve2d(img1, kernel_x, boundary='symm', mode='same')
    fy = signal.convolve2d(img1, kernel_y, boundary='symm', mode='same')

    # TODO: fix ft, it now gives only zeros

    ft = signal.convolve2d(img2, kernel_t, boundary='symm', mode='same') + \
         signal.convolve2d(img1, -kernel_t, boundary='symm', mode='same')

    # Get values within window size distance
    w = block_shape[0]

    # Initialize starting positions.
    X = np.zeros(img1.shape)
    Y = np.zeros(img1.shape)

    v = np.zeros(img1.shape)
    print(v)
    u = np.zeros(img1.shape)

    for i in range(w):
        for j in range(w):

            # Get derivatives of the current region
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            # Get A and b
            A = np.column_stack((Ix, Iy))
            # print(A)
            Atrans = np.transpose(A)
            b = -It

            # Get optical flow by: v = (A^TA)^−1 A^Tb
            AtA = np.dot(Atrans, A)
            # print(AtA)

            X[i, j] = i
            Y[i, j] = j

            if np.linalg.det(AtA) != 0:
                np.linalg.inv(AtA)
                inverse = np.linalg.inv(AtA)
                Atb = np.dot(Atrans, b)
                equation_solution = np.dot(inverse, Atb)
                # print(equation_solution)
                v[i, j] = equation_solution[1]
                u[i, j] = equation_solution[0]
                # print(v)
            else:
                u[i, j] = 0
                v[i, j] = 0

    plot_quiver(X, Y, u, v)


def plot_quiver(X, Y, u, v):
    # TODO: figure out right input for quiver
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.quiver(X, Y, u, v)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    # ax.axis([-0.2, 2.3, -0.2, 2.3])
    ax.set_aspect('equal')

    plt.show()


if __name__ == "__main__":
    sphere_1 = plt.imread('sphere1.ppm')
    print(sphere_1.shape)
    sphere_2 = plt.imread('sphere1.ppm')
    print(sphere_2.shape)

    # synth_1 = plt.imread('synth1.pgm')
    # print(synth_1.shape)
    # synth_2 = plt.imread('synth2.pgm')
    # print(synth_2.shape)

    get_nonoverlapping_regions(sphere_1, sphere_2)