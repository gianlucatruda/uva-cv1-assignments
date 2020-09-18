from gauss2D import gauss2D
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt


def calculate_laplacian(image):
    kernel = np.array([1, -2, 1]).reshape(1, 3)
    dx = convolve2d(image, kernel, mode='same')
    dy = convolve2d(image.T, kernel, mode='same').T
    return dx+dy


def compute_LoG(image, LOG_type, show_intermediate=False):
    sigma = 0.5
    kernel_size = 5
    if LOG_type == 1:
        # First gaussian, then Laplacian of the image
        smoothed = convolve2d(image, gauss2D(sigma, kernel_size))
        res = calculate_laplacian(smoothed)
        if show_intermediate:
            plt.subplot(1, 2, 1)
            plt.imshow(smoothed, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(res, cmap='gray')
            plt.show()
    elif LOG_type == 2:
        # Laplacian of Gaussian kernel
        lod = calculate_laplacian(gauss2D(sigma, kernel_size))
        res = convolve2d(image, lod)
        if show_intermediate:
            plt.subplot(1, 2, 1)
            plt.imshow(lod, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(res, cmap='gray')
            plt.show()
    elif LOG_type == 3:
        # Difference of Gaussians
        gauss_narrow = gauss2D(sigma, kernel_size)
        gauss_wide = gauss2D(10*sigma, kernel_size)
        res = convolve2d(image, gauss_narrow - gauss_wide)
    return res


if __name__=='__main__':
    image = plt.imread('images/image2.jpg')
    res = compute_LoG(image, 1)
    plt.imshow(res, cmap='gray')
    plt.show()