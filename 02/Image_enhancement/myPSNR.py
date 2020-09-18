import numpy as np
import math
import cv2

def myPSNR(orig_image, approx_image ):

    I_MAX = 255

    or_image = orig_image.astype(np.uint8)
    app_image = approx_image.astype(np.uint8)
    mse = np.mean((or_image - app_image) ** 2)
    if mse == 0:
        return float('inf')
    PSNR = 20 * math.log((I_MAX / math.sqrt(mse)), 10)
    return PSNR


if __name__ == "__main__":
    img1_path = 'images/image1.jpg'
    img2_path = 'images/image1_saltpepper.jpg'
    img3_path = 'images/image1_gaussian.jpg'

    # Read with opencv
    orig_image = cv2.imread(img1_path)
    # approx_image = cv2.imread(img2_path)
    approx_image = cv2.imread(img3_path)
    PSNR = myPSNR(orig_image, approx_image)
    print(PSNR)

