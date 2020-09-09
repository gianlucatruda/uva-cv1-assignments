from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

"""
The algorithm assumes that, under a white light source, the average colour in a scene should be achromatic (grey, [128, 128, 128]).

You do not need to apply any pre or post processing steps.

For the calculation or processing, you are not allowed to use any available code or any dedicated library function except standard Numpy functions .
"""


def greyworld_correction(img):
    _img = img.copy()
    original_shape = _img.shape
    M, N = original_shape[0], original_shape[1]

    # Reshape image to (3, MN) — list of RGB 3-tuples, tranposed
    _img = _img.reshape(M*N, 3).T
    print(_img.shape)

    # Convert RBG to xyz (LMS) space with a transformation matrix (von Kries)
    # https://ixora.io/projects/colorblindness/color-blindness-simulation-research/
    B = np.array(
        [[0.31399022, 0.15537241, 0.01775239],
         [0.63951294, 0.75789446, 0.10944209],
         [0.04649755, 0.08670142, 0.87256922]])

    # Apply the RGB -> LMS transformation to iamge using python's dot operator
    _img = B @ _img

    # Calculate the average colour value for each channel
    mean_rgb = np.mean(_img, axis=1)

    f = 2.0  # Scaling factor that depends on scene
    # Loop through the RGB values and apply greyworld adjustments (divide RGB by mean RGB)
    adj_img = []
    for pixel in _img.T:
        adj_img.append([
            pixel[0] / (f * mean_rgb[0]),
            pixel[1] / (f * mean_rgb[1]),
            pixel[2] / (f * mean_rgb[2]),
        ])

    # Convert the list of output pixels to an array and transpose
    adj_img = np.array(adj_img).T

    # Reshape into a 2D image
    adj_img = adj_img.T.reshape(original_shape)

    return adj_img


if __name__ == "__main__":
    IMG_PATH = Path('awb.jpg')

    # Read the image
    img = plt.imread(IMG_PATH)
    print(f"{IMG_PATH}: {img.shape}")
    conv_img = greyworld_correction(img)

    # Calculate average pixel values for both images
    M, N = img.shape[0], img.shape[1]
    mean_rgb_orig = np.mean(img.reshape(M*N, 3).T, axis=1)
    mean_rgb_conv = np.mean(conv_img.reshape(M*N, 3).T, axis=1)
    print(mean_rgb_orig, mean_rgb_conv)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[1].imshow(conv_img)

    plt.show()
