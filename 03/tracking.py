"""
1. Implement a simple feature-tracking algorithm by following below steps. Name your script tracking.py.

    (a) Locate feature points on the first frame by using the Harris Corner Detector, that you implemented in section 1.

    (b) Track these points using the Lucas-Kanade algorithm for optical flow estimation, that you implemented in section 2.

2. Prepare a video for each sample image sequences. These videos should visualize the initial feature points and the optical flow. Test your imple - mentation and prepare visualization videos for pingpong and person toy samples.
"""

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import cv2

from harris_corner_detector import calculate_harris, plot_grad
from lucas_kanade import lucas_kanade, plot_vector_field


if __name__ == '__main__':
    DIR = 'person_toy'
    # DIR = 'pingpong'

    # Read in all images in directory
    frames = []
    frame_path = Path(DIR)
    for img in frame_path.glob('*.jp*g'):
        frames.append(plt.imread(img))

    nframes = len(frames)
    rows, cols = frames[0].shape[0], frames[0].shape[1]

    # Find Harris corners in first frame
    H, r, c = calculate_harris(frames[0], sigma=3, window=3, thr=0.01)
    # plot_grad(frames[0], r, c)

    # Construct feature image
    features = np.zeros((rows, cols))
    for x, y in zip(c, r):
        features[x, y] = 1

    # Track Harris corners through subsequent frames
    for i, f in enumerate(frames[1:5]):
        print("frame:", i)

        # Run LK on features
        V = lucas_kanade(f, frames[i+1])
        Vx, Vy = V[..., 0], V[..., 1]

        import ipdb; ipdb.set_trace()

        # Update features by IxVx + IyVy = −It
        features = -1 @ (features @ Vx + features @ Vy)

        plt.imshow(f)
        plt.imshow(features)
        plt.show()

    # TODO make video