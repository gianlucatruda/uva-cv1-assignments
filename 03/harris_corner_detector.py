from scipy.signal import convolve2d
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt
import cv2


def calculate_harris(img, sigma=3, max_window=3, thr=0):
    img2d = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Ix = convolve2d(img2d, np.array([[-1, 0, 1]]), 'same')  # gradient by x-axis
    Iy = convolve2d(img2d.T, np.array([[-1, 0, 1]]), 'same').T  # gradient by y-axis
    A = gaussian_filter(Ix**2, sigma)
    B = gaussian_filter(Ix*Iy, sigma)
    C = gaussian_filter(Iy**2, sigma)
    H = (A * C - B**2) - 0.04*(A + C)**2  # cornerness matrix

    # searching for local maxima in H to detect corners
    r, c = np.where((maximum_filter(H, max_window) == H) & (maximum_filter(H, max_window) >= thr))
    return H, r, c


if __name__=='__main__':
    img = plt.imread('person_toy/00000001.jpg')
    H,r,c = calculate_harris(img, sigma=1, thr=1)
    print(f'There are {r.shape[0]} corner points')
    plt.imshow(H, cmap='gray')
    plt.show()