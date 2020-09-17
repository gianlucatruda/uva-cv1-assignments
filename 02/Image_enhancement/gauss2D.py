import numpy as np
from gauss1D import gauss1D

def gauss2D(sigma, kernel_size):
    raw = gauss1D(sigma, kernel_size) * gauss1D(sigma, kernel_size).reshape(-1,1)
    G = raw / np.sum(raw)
    return G


if __name__ == '__main__':
    print(gauss2D(2, 5))

    import matplotlib.pyplot as plt
    import cv2
    from scipy.signal import convolve2d

    img = plt.imread('preview-11.png')
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    gaussed = convolve2d(img, gauss2D(5, 11))
    print(img.shape)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(gaussed)
    plt.show()