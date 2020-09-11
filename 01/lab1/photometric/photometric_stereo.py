import numpy as np
import cv2
import os
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface

print('Part 1: Photometric Stereo\n')


def photometric_stereo(image_dir='./SphereGray5/', colour=False):

    # obtain many images in a fixed view under different illumination
    print('Loading images...\n')
    if colour:
        print('Stacking across colour channels...\n')
        img_stacks, Vs = [], []
        for i in range(3):
            [image_stack, scriptV] = load_syn_images(image_dir, channel=i)
            img_stacks.append(image_stack)
            Vs.append(scriptV)

        # Reorder from CV2's BGR to RGB
        img_stacks = [img_stacks[2], img_stacks[1], img_stacks[0]]
        image_stack = np.stack(img_stacks, axis=-1)
        scriptV = np.stack(Vs, axis=-1)
        [h, w, n, c] = image_stack.shape
    else:
        print('Assuming single-channel images (grayscale).\n')
        [image_stack, scriptV] = load_syn_images(image_dir, channel=0)
        [h, w, n] = image_stack.shape

    print('Finish loading %d images.\n' % n)

    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    [albedo, normals] = estimate_alb_nrm(
        image_stack, scriptV, shadow_trick=False, colour=colour)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking\n')
    [p, q, SE] = check_integrability(normals)

    threshold = 0.005
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan')  # for good visualization

    # compute the surface height
    height_map = construct_surface(p, q, path_type='average')

    # show results
    show_results(albedo, normals, height_map, SE)


def photometric_stereo_face(image_dir='./yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan')  # for good visualization

    # compute the surface height
    height_map = construct_surface(p, q)

    # show results
    show_results(albedo, normals, height_map, SE)


if __name__ == '__main__':
    photometric_stereo('photometrics_images/SphereGray5/', colour=False)
    photometric_stereo('photometrics_images/SphereColor/', colour=True)
    photometric_stereo('photometrics_images/MonkeyGray/', colour=False)
    photometric_stereo('photometrics_images/MonkeyColor/', colour=True)
    photometric_stereo_face('photometrics_images/yaleB02')
