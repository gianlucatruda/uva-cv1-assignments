import numpy as np
import cv2
from getColourChannels import *

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods

    R, G, B = getColourChannels(input_image)


    light = (input_image.max(axis=2) + input_image.min(axis=2)) / 2
    new_image_l = np.zeros(input_image.shape, dtype='uint8')
    new_image_l[:, :, 0], new_image_l[:, :, 1], new_image_l[:, :, 2] = light, light, light


    # average method
    average = (R + G + B) / 3
    new_image_a = np.zeros(input_image.shape, dtype='uint8')
    new_image_a[:, :, 0], new_image_a[:, :, 1], new_image_a[:, :, 2] = average, average, average

    # luminosity method
    luminosity = 0.21 * R + 0.72 * G + 0.07 * B
    new_image_lu = np.zeros(input_image.shape, dtype='uint8')
    new_image_lu[:, :, 0], new_image_lu[:, :, 1], new_image_lu[:, :, 2] = luminosity, luminosity, luminosity


    # built-in opencv function
    new_image = cv2.cvtColor(input_image, code=cv2.COLOR_RGB2GRAY)

    return new_image_l, new_image_a, new_image_lu, new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space

    R, G, B = getColourChannels(input_image)

    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2 * B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)

    new_image = np.zeros(input_image.shape, dtype='uint8')
    new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2] = O1, O2, O3

    return new_image


def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space

    R, G, B = getColourChannels(input_image)
    rgb_norm = np.zeros(input_image.shape, dtype="uint8")

    sum = R + G + B

    # Black pixels result in division by 0, so take nonzero number.
    sum[sum == 0] = 0.01
    rgb_norm[:, :, 0] = R / sum * 255
    rgb_norm[:, :, 1] = G / sum * 255
    rgb_norm[:, :, 2] = B / sum * 255

    new_image = rgb_norm


    return new_image
