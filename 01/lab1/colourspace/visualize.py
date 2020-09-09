from PIL import Image
import numpy as np
import cv2


def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    print(input_image.shape)
    new_im = input_image.astype(np.uint8)
    print(new_im.shape)
    cv2.imwrite('test_gray_light.jpeg', new_im)
    # new_im.save('test_opponent.jpeg')
