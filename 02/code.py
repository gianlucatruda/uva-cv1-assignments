from math import pi
import numpy as np
from matplotlib import pyplot as plt


def gauss1D(sigma, kernel_size):
    # your code
    return G


def gauss2D(sigma, kernel_size):
    # your code
    return G


def createGabor(lamb, theta, psi, sigma, gamma):

    height, width = 100, 100

    def gaussian(x_prime, y_prime, sigma, gamma):
        return np.exp(-(x_prime ** 2 + (gamma ** 2 * y_prime ** 2)) / (2 * sigma ** 2))

    def carrier(x_prime, lamb, psi):
        return 2 * pi * (x_prime / lamb) + psi

    def gabor(x, y, lamb, theta, psi, sigma, gamma):
        x_prime = x * np.cos(theta) + y * np.sin(theta)
        y_prime = -x * np.sin(theta) + y * np.cos(theta)
        g_real = gaussian(x_prime, y_prime, sigma, gamma) * np.cos(carrier(x_prime, lamb, psi))
        g_im = gaussian(x_prime, y_prime, sigma, gamma) * np.sin(carrier(x_prime, lamb, psi))

        return g_real, g_im

    real = np.zeros((width, height))
    im = np.zeros((width, height))

    for x in range(-width // 2, width // 2):
        for y in range(-height // 2, height // 2):
            xoff, yoff = width // 2, height // 2
            real[x+xoff, y+yoff], im[x+xoff, y+yoff] = gabor(x, y, lamb, theta, psi, sigma, gamma)

    return real, im


if __name__ == "__main__":
    real, im = createGabor(15, 0, 0, 30, 1)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(real, cmap="gray")
    ax[1].imshow(im, cmap="gray")
    plt.show()
