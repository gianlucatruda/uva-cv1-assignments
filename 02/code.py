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
        g_real = gaussian(x_prime, y_prime, sigma, gamma) * \
            np.cos(carrier(x_prime, lamb, psi))
        g_im = gaussian(x_prime, y_prime, sigma, gamma) * \
            np.sin(carrier(x_prime, lamb, psi))

        return g_real, g_im

    real = np.zeros((width, height))
    im = np.zeros((width, height))

    for x in range(-width // 2, width // 2):
        for y in range(-height // 2, height // 2):
            xoff, yoff = width // 2, height // 2
            real[x+xoff, y+yoff], im[x+xoff, y +
                                     yoff] = gabor(x, y, lamb, theta, psi, sigma, gamma)

    return real, im


if __name__ == "__main__":
    fig, ax = plt.subplots(3, 5)

    cmap = 'gray'

    # Theta
    for i, val in enumerate([0, 1, 2, 3, 4]):
        real, im = createGabor(15, pi*val/4, 0, 20, 1)
        ax[0][i].imshow(real, cmap=cmap)
        ax[0][i].set_title(r"$\theta$="+str(val)+r"$\frac{\pi}{4}$")
        ax[0][i].axis('off')

    # Sigma
    for i, val in enumerate([3, 10, 20, 30, 50]):
        real, im = createGabor(15, pi/2, 0, val, 1)
        ax[1][i].imshow(real, cmap=cmap)
        ax[1][i].set_title(r"$\sigma =$"+str(val))
        ax[1][i].axis('off')

    # Gamma
    for i, val in enumerate([0.1, 0.5, 1.0, 1.5, 10.0]):
        real, im = createGabor(15, pi/2, 0, 20, val)
        ax[2][i].imshow(real, cmap=cmap)
        ax[2][i].set_title(r"$\gamma =$"+str(val))
        ax[2][i].axis('off')

    plt.tight_layout()
    plt.show()
