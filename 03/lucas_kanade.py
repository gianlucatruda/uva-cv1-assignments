from cv2 import imread
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def lucas_kanade(image_1, image_2):
    """ Run Lucas Kanade on a pair of images using 15x15 windows.
    """

    # Convert to grayscale by averaging channels (if necessary)
    if image_1.ndim == 3:
        image_1 = np.mean(image_1, axis=2)
    if image_2.ndim == 3:
        image_2 = np.mean(image_2, axis=2)

    # Motion array initialised
    V_img = np.zeros((image_1.shape[0], image_1.shape[1], 2))
    rows = image_1.shape[0]
    windows = rows // 15  # TODO remove hardcoding

    # Loop over windows
    for i in range(0, 15*windows, 15):
        for j in range(0, 15*windows, 15):
            block_1 = image_1[i:i+15, j:j+15]
            block_2 = image_2[i:i+15, j:j+15]

            # Apply lucas kanade on that window
            V = lk_on_window(block_1, block_2)
            V_img[i:i+15, j:j+15] = V

    return V_img


def lk_on_window(img1, img2):
    """ Run Lucas Kanade on a pair of windows
    """

    # Prewitt filter approach
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])

    fx = signal.convolve2d(img1, kernel_x, boundary='symm', mode='same')
    fy = signal.convolve2d(img1, kernel_y, boundary='symm', mode='same')
    ft = signal.convolve2d(img1, kernel_t, boundary='symm', mode='same') + \
        signal.convolve2d(img2, -1 * kernel_t, boundary='symm', mode='same')

    # Get values within window
    w = img1.shape[0]

    # Initialize flow matrix
    V = np.zeros((w, w, 2))

    # Get A and b
    A = np.column_stack((fx.flatten(), fy.flatten()))
    b = -1 * ft.flatten()

    # Get optical flow by: v = (A^TA)^−1 A^Tb
    V, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return V


def plot_vector_field(img, V, ax, step=10, scale=None):
    """ Plot overlayed vector fields
    """

    rows = img.shape[0]
    cols = img.shape[1]

    Vx = V[::step, ::step, 0]
    Vy = V[::step, ::step, 1]
    X, Y = np.arange(0, rows, step), np.arange(0, cols, step)
    ax.imshow(img, cmap='gray', alpha=0.5)
    ax.quiver(X, Y, Vx, Vy, angles='xy', color='red',
              scale_units='x', scale=scale, headwidth=3, width=0.005)

    return ax


if __name__ == "__main__":

    fig, ax = plt.subplots(1, 2, dpi=200)

    sphere_1 = plt.imread('sphere1.ppm')
    sphere_2 = plt.imread('sphere2.ppm')
    V1 = lucas_kanade(sphere_1, sphere_2)
    plot_vector_field(sphere_1, V1, ax[0], scale=1)

    synth_1 = plt.imread('synth1.pgm')
    synth_2 = plt.imread('synth2.pgm')
    V2 = lucas_kanade(synth_1, synth_2)
    plot_vector_field(synth_1, V2, ax[1])
    plt.tight_layout()
    plt.show()
