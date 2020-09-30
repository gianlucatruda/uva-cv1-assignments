import cv2
import os
import random


def match_keypoints(im1, im2,
                    show_keypoints=False,
                    show_matches=True,
                    ratio=0.2,
                    limit=None):
    # New API https://stackoverflow.com/questions/18561910/cant-use-surf-sift-in-opencv#32735795
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    coords = []
    for match in good:
        p1 = kp1[match[0].queryIdx].pt
        p2 = kp2[match[0].trainIdx].pt
        coords.append((p1, p2))
        # print(p1, p2)

    if show_matches:

        # Apply optional limit and take random subset
        random.shuffle(good)
        good = good[:limit] if limit is not None else good

        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(
            im1, kp1, im2, kp2, good, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('matching', img3)
        cv2.imwrite('figs/matches.png', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if show_keypoints:
        img1 = cv2.drawKeypoints(
            im1, kp1, im1,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2 = cv2.drawKeypoints(
            im2, kp2, im2,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image1', img1)
        cv2.imshow('image2', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return coords, kp1, kp2


if __name__ == '__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')

    if not os.path.exists('figs'):
        os.makedirs('figs')

    matches, _, _ = match_keypoints(im1, im2,
                                    show_matches=True,
                                    limit=10,
                                    ratio=0.2)
    print(f"Found {len(matches)} matches")
