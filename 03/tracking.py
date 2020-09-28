
from pathlib import Path
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy import signal

from harris_corner_detector import calculate_harris, plot_grad
from lucas_kanade import lucas_kanade, lk_on_window, plot_vector_field


if __name__ == '__main__':
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
    H, c, r = calculate_harris(frames[0], sigma=3, window=3, thr=0.01)

    # Ditch any points on edges of image (common)
    _c, _r = [], []
    for x, y in zip(c, r):
        if x > 50 and y > 50:
            _c.append(x)
            _r.append(y)
    c, r = _c, _r
    print(f"Tracking {len(c)} points")

    # plot_grad(frames[0], c, r)

    # Sanity check about ordering of rows and columns
    # print('R:', r)
    # print('C:', c)

    image_frames = []

    # Track Harris corners through subsequent frames
    for i, f in enumerate(frames[:-1]):
        print(f"Frame: {i}/{len(frames)}", end='\r')

        # Convert to grayscale
        frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        next_frame = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)

        # Run LK on features
        _c, _r = [], []
        img = frames[i].copy()

        for x, y in zip(c, r):

            # Create windows (N x N) around each Harris point
            N = 20 // 2
            winx_1, winx_2 = max(0, x-N), min(cols, x+N)
            winy_1, winy_2 = max(0, y-N), min(rows, y+N)
            block_1 = frame[winx_1:winx_2, winy_1:winy_2]
            block_2 = next_frame[winx_1:winx_2, winy_1:winy_2]

            # Apply lucas kanade on those windows
            V = lk_on_window(block_1, block_2)
            Vx, Vy = V[..., 0], V[..., 1]

            # Update features
            _x = x + 10*Vx
            _y = y + 10*Vy
            _c.append(round(_x))
            _r.append(round(_y))

            # Visualise Harris points, LK windows, motion vectors
            img = cv2.rectangle(img, (winx_1, winy_1), (winx_2, winy_2), (255, 0, 0), 1)
            img = cv2.arrowedLine(img, (x, y), (round(x + 50*Vx), round(y + 50*Vy)),color=(0, 255, 0), thickness=2)
            img = cv2.circle(img, center=(x, y), radius=1, color=(0, 0, 255), thickness=2)

        # Save the updated points
        c, r = _c, _r

        # Save the composite image
        cv2.imwrite(f"{savepath}/{str(i).zfill(5)}.png", img)
        image_frames.append(img)


    # Make video of all images
    out = cv2.VideoWriter(f"{savepath}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (cols, rows))
    for i in range(len(image_frames)):
        out.write(image_frames[i])
    out.release()
