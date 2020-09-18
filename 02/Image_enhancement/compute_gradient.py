from scipy.signal import convolve2d
import numpy as np

def compute_gradient(image):
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gx = convolve2d(sobel, image)
    Gy = convolve2d(sobel.T, image)
    im_magnitude = np.sqrt(Gx**2 + Gy**2)
    im_direction = np.arctan(Gy/Gx)
    return Gx, Gy, im_magnitude, im_direction


if __name__=='__main__':
    import matplotlib.pyplot as plt
    Gx, Gy, magn, dir = compute_gradient(plt.imread('images/image1.jpg'))
    plt.imshow(Gx)
    plt.show()
    plt.imshow(Gy)
    plt.show()
    plt.imshow(magn)
    plt.show()
    plt.imshow(dir)
    plt.show()