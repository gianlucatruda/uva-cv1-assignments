
from pathlib import Path
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy import signal

from harris_corner_detector import calculate_harris, plot_grad
from lucas_kanade import lucas_kanade, lk_on_window, plot_vector_field


def grayscale(img: np.ndarray):
    # Convert to grayscale by averaging channels (if necessary)
    if img.ndim == 3:
        return np.mean(img, axis=2)
    return img


if __name__ == '__main__':
    SCALER = 5
    DIR = 'person_toy'
    DIR = 'pingpong'
    OUTDIR = 'output'

    savepath = Path(OUTDIR) / DIR
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Read in all images in directory
    frames = []
    frame_path = Path(DIR)
    for img in sorted(list(frame_path.glob('*.jp*g'))):
        frames.append(cv2.imread(str(img)))

    nframes = len(frames)
    rows, cols = frames[0].shape[0], frames[0].shape[1]

    # Find Harris corners in first frame
    H, c, r = calculate_harris(frames[0], sigma=0.8, window=10, thr=0.005)

    # Ditch any points on edges of image (common)
    _c, _r = [], []
    for x, y in zip(c, r):
        if x > 10 and y > 10:
            _c.append(x)
            _r.append(y)
    c, r = _c, _r
    print(f"Tracking {len(c)} points")

    # plot_grad(frames[0], c, r)

    # Sanity check about ordering of rows and columns
    # print('R:', r)
    # print('C:', c)

    # Track Harris corners through subsequent frames
    for i, f in enumerate(frames[:-1]):
        print(f"Frame: {i}/{len(frames)}", end='\r')

        # Convert to grayscale
        frame = grayscale(frames[i])
        next_frame = grayscale(frames[i+1])

        # Run LK on features
        _c, _r = [], []
        img = frames[i].copy()

        for x, y in zip(c, r):

            # Create windows (20x20) around each Harris point
            winx_1, winx_2 = max(0, x-10), min(cols, x+10)
            winy_1, winy_2 = max(0, y-10), min(rows, y+10)
            block_1 = frame[winx_1:winx_2, winy_1:winy_2]
            block_2 = next_frame[winx_1:winx_2, winy_1:winy_2]

            # Apply lucas kanade on those windows
            V = lk_on_window(block_1, block_2)
            Vx, Vy = V[..., 0], V[..., 1]

            # Update features
            _x = x + SCALER*Vx
            _y = y + SCALER*Vy
            _c.append(int(_x))
            _r.append(int(_y))

            # Visualise Harris points, LK windows, motion vectors
            img = cv2.rectangle(img, (winx_1, winy_1),
                                (winx_2, winy_2), (255, 0, 0), 1)
            img = cv2.arrowedLine(
                img, (x, y), (int(x + 30*Vx), int(y + 30*Vy)),
                color=(0, 255, 0), thickness=1)
            img = cv2.circle(img, center=(x, y), radius=2,
                             color=(0, 0, 255), thickness=2)

        # Save the updated points
        c, r = _c, _r

        # Save the composite image
        cv2.imwrite(f"{savepath}/{str(i).zfill(5)}.png", img)

    # TODO make video
