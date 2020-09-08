from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

"""
The algorithm assumes that, under a white light source, the average colour in a scene should be achromatic (grey, [128, 128, 128]).

You do not need to apply any pre or post processing steps.

For the calculation or processing, you are not allowed to use any available code or any dedicated library function except standard Numpy functions .
"""


def srgb_to_linsrgb(srgb):
    """Convert sRGB values to physically linear ones. The transformation is
       uniform in RGB, so *srgb* can be of any shape.

    """
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    return np.where(srgb > 0.04045, gamma, scale)


def greyworld_correction(img):
    _img = img.copy()
    original_shape = _img.shape
    M, N = original_shape[0], original_shape[1]

    # Reshape image to (3, MN) — list of RGB 3-tuples, tranposed
    _img = _img.reshape(M*N, 3).T
    print(_img.shape)

    # Linearise the JPG input
    # _img = np.array(list(map(srgb_to_linsrgb, _img)))
    print(_img.shape)

    # Convert RBG to xyz (LMS) space with a transformation matrix (von Kries)
    A = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])

    # Apply the RGB -> xyz transformation to iamge using python's dot operator
    _img = A @ _img

    # Calculate the average colour value for each channel
    mean_rgb = np.mean(_img, axis=1)

    # Loop through the RGB values and apply greyworld adjustments (divide RGB by mean RGB)
    adj_img = []
    for pixel in _img.T:
        adj_img.append([
            pixel[0] / mean_rgb[0],
            pixel[1] / mean_rgb[1],
            pixel[2] / mean_rgb[2],
        ])

    # Convert the list of output pixels to an array and transpose
    adj_img = np.array(adj_img).T

    # Map back to RGB colourspace for output using inverse of von Kries matrix
    adj_img = np.linalg.inv(A) @ adj_img

    # Reshape into a 2D image
    adj_img = adj_img.T.reshape(original_shape)

    return adj_img


if __name__ == "__main__":
    IMG_PATH = Path('awb.jpg')

    # Read the image
    img = plt.imread(IMG_PATH)
    print(f"{IMG_PATH}: {img.shape}")
    conv_img = greyworld_correction(img)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[1].imshow(conv_img)

    plt.show()
