from scipy.signal import convolve2d
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt
import cv2


def calculate_harris(img, sigma=3, max_window=3, thr=0.1):
    img2d = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Ix = convolve2d(img2d, np.array([[-1, 0, 1]]), 'same')  # gradient by x-axis
    Iy = convolve2d(img2d.T, np.array([[-1, 0, 1]]), 'same').T  # gradient by y-axis
    A = gaussian_filter(Ix**2, sigma)
    B = gaussian_filter(Ix*Iy, sigma)
    C = gaussian_filter(Iy**2, sigma)
    H = (A * C - B**2) - 0.04*(A + C)**2  # cornerness matrix

    # searching for local maxima in H to detect corners
    r, c = np.where((maximum_filter(H, max_window) == H) & (maximum_filter(H, max_window) >= thr*np.max(H)))
    return H, r, c


def plot_grad(img, r, c):
    img2d = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Ix = convolve2d(img2d, np.array([[-1, 0, 1]]), 'same')  # gradient by x-axis
    Iy = convolve2d(img2d.T, np.array([[-1, 0, 1]]), 'same').T  # gradient by y-axis
    plt.subplot(2, 2, 1)
    plt.imshow(Ix, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(Iy, cmap='gray')
    plt.subplot2grid((2,2), (1,0), colspan=2)
    for point in list(zip(c,r)):
        img = cv2.circle(img, center=point, radius=10, color=(0,0,255), thickness=-1)
    plt.imshow(img)
    plt.show()


if __name__=='__main__':
    #fname = 'person_toy/00000003.jpg'
    fname = 'pingpong/0000.jpeg'
    img = np.array(plt.imread(fname))
    img.setflags(write=1)
    H,r,c = calculate_harris(img, max_window=4, sigma=3, thr=0.05)
    print(f'There are {r.shape[0]} corner points')
    plot_grad(img, r, c)
    