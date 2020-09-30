import cv2
import numpy as np
import matplotlib.pyplot as plt
from myPSNR import *


def denoise(image, kernel_type, kernel_size):
    image_ready = image.astype(np.uint8)

    if kernel_type == 'box':
        imOut = cv2.blur(image_ready, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    elif kernel_type == 'median':
        imOut = cv2.medianBlur(image_ready, kernel_size)
    elif kernel_type == 'gaussian':
        imOut = cv2.GaussianBlur(image_ready, (kernel_size, kernel_size), 0)
    else:
        print('Operatio Not implemented')
    return imOut


if __name__ == "__main__":
    # Get all possible combinations to test.
    # img_paths = ['images/image1_saltpepper.jpg', 'images/image1_gaussian.jpg']
    img_paths = ['images/image1_gaussian.jpg']

    # kernel_types = ['box', 'median']
    kernel_types = ['gaussian']
    kernel_sizes = [3, 5, 7]
    std_dev = [0, 0.25, 0.5, 1, 5]

    fig = plt.figure()
    num = 0

    for img_path in img_paths:
        image = cv2.imread(img_path)

        for dev in std_dev:
            for kernel_type in kernel_types:
                for kernel_size in kernel_sizes:
                    imOut = denoise(image, kernel_type, kernel_size)

                    PSNR = myPSNR(image, imOut)

                    num += 1
                    ax = fig.add_subplot(3, 5, num)
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)
                    ax.imshow(imOut, interpolation='bilinear', origin='upper', extent=[-3, 3, -3, 3])
                    # ax.set_title(str(kernel_type) + " " + str(kernel_size) + " " + str(img_path[14]))
                    ax.set_title(str(kernel_size) + " std_" + str(dev))
                    print(str(kernel_type) + " " + str(kernel_size) + " " + str(img_path[6:] + " gives value:"))
                    print(PSNR)


    plt.tight_layout()
    plt.savefig("gaussian.jpeg", dpi=300)
    plt.show()



