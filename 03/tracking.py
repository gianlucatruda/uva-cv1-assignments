"""
1. Implement a simple feature-tracking algorithm by following below steps. Name your script tracking.py.

    (a) Locate feature points on the first frame by using the Harris Corner Detector, that you implemented in section 1.

    (b) Track these points using the Lucas-Kanade algorithm for optical flow estimation, that you implemented in section 2.

2. Prepare a video for each sample image sequences. These videos should visualize the initial feature points and the optical flow. Test your imple - mentation and prepare visualization videos for pingpong and person toy samples.
"""

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from harris_corner_detector import calculate_harris
from lucas_kanade import lucas_kanade


if __name__ == '__main__':
    DIR = 'person_toy'
    # DIR = 'pingpong'

    # Read in all images in directory
    frames = []
    frame_path = Path(DIR)
    for img in frame_path.glob('*.jp*g'):
        frames.append(plt.imread(img))

    # Stack images into a single 4D array (x, y, channel, time)
    nframes = len(frames)
    rows, cols = frames[0].shape[0], frames[0].shape[1]
    vid = np.stack(frames, axis=3)
    print(vid.shape)

    # Find Harris corners in first frame
    # TODO

    # Track Harris corners through subsequent frames
    # TODO

    # Visualise
    # TODO
