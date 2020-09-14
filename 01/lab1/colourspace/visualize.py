from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib


def visualize(input_image):
    new_im = input_image.astype(np.uint8)

    # For visualization of gray:
    # new_im_l, new_im_a, new_im_lu, new_im = input_image
    # new_im_l, new_im_a, new_im_lu, new_im = new_im_l.astype(np.uint8), new_im_a.astype(np.uint8), \
    #                                         new_im_lu.astype(np.uint8), new_im.astype(np.uint8)

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(new_im, interpolation='bilinear', cmap='gray', origin='upper', extent=[-3, 3, -3, 3])
    ax[0, 0].set_axis_off()
    ax[0, 0].set_title("Total")
    ax[0, 1].imshow(new_im[:, :, 0], interpolation='bilinear', cmap='gray', origin='upper', extent=[-3, 3, -3, 3])
    ax[0, 1].set_axis_off()
    ax[0, 1].set_title("Luminance")
    ax[1, 0].imshow(new_im[:, :, 1], interpolation='bilinear', cmap='gray', origin='upper', extent=[-3, 3, -3, 3])
    ax[1, 0].set_axis_off()
    ax[1, 0].set_title("Chroma blue")
    ax[1, 1].imshow(new_im[:, :, 2], interpolation='bilinear', cmap='gray', origin='upper', extent=[-3, 3, -3, 3])
    ax[1, 1].set_axis_off()
    ax[1, 1].set_title("Chroma red")

    plt.savefig("test_ycbcr.jpeg")
    # matplotlib.imwrite('test_gray_light.jpeg', new_im)
    # new_im.save('test_opponent.jpeg')
