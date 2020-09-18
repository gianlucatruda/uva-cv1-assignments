import numpy as np

def gauss(x,sigma):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2))

def gauss1D(sigma, kernel_size):
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    # solution
    raw = gauss(np.arange(int(-kernel_size / 2), int(kernel_size / 2) + 1), sigma)
    G = raw / np.sum(raw)
    return G

if __name__ == '__main__':
    print(gauss1D(2,5))

    import matplotlib.pyplot as plt
    sin = np.sin(np.linspace(-10,10,200)) + np.random.normal(0, 0.3, 200)
    gaussed = np.convolve(sin, gauss1D(2, 5))
    plt.subplot(1,2,1)
    plt.plot(sin)
    plt.subplot(1,2,2)
    plt.plot(gaussed)
    plt.show()