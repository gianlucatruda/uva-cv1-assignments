import cv2


def match_keypoints(im1, im2, ratio=0.15, show_keypoints=False, show_matches=True):
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
        print(p1, p2)

    if show_matches:
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(
            im1, kp1, im2, kp2, good, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('matching', img3)
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

    matches, kp1, kp2 = match_keypoints(im1, im2, ratio=0.15, show_matches=True)

    print(f"Found {len(matches)} matches")