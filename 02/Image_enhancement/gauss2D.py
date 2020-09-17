import numpy as np
from gauss1D import gauss1D

def gauss2D(sigma, kernel_size):
    raw = gauss1D(sigma, kernel_size) * gauss1D(sigma, kernel_size).reshape(-1,1)
    G = raw / np.sum(raw)
    return G


if __name__ == '__main__':
    print(gauss2D(2, 5))