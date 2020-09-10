from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib


def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    print(input_image.shape)
    new_im = input_image.astype(np.uint8)
    print(new_im.shape)
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(new_im, interpolation='bilinear', cmap='gray', origin='upper', extent=[-3, 3, -3, 3])
    ax[0, 0].set_axis_off()
    ax[0, 0].set_title("Grey")
    ax[0, 1].imshow(new_im[:, :, 0], interpolation='bilinear', cmap='gray', origin='upper', extent=[-3, 3, -3, 3])
    ax[0, 1].set_axis_off()
    ax[0, 1].set_title("same?")
    ax[1, 0].imshow(new_im[:, :, 1], interpolation='bilinear', cmap='gray', origin='upper', extent=[-3, 3, -3, 3])
    ax[1, 0].set_axis_off()
    ax[1, 0].set_title("same?")
    ax[1, 1].imshow(new_im[:, :, 2], interpolation='bilinear', cmap='gray', origin='upper', extent=[-3, 3, -3, 3])
    ax[1, 1].set_axis_off()
    ax[1, 1].set_title("same?")

    plt.savefig("test_gray_average.jpeg")
    # matplotlib.imwrite('test_gray_light.jpeg', new_im)
    # new_im.save('test_opponent.jpeg')
