import numpy as np


def estimate_alb_nrm(image_stack, scriptV, shadow_trick=True):

    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w, _ = image_stack.shape

    # create arrays for
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])

    """
    ================
    Your code here
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point
        albedo at this point is |g|
        normal at this point is g / |g|
    """

    # for each point in image_stack array
    for x in range(h):
        for y in range(w):

            # stack image values into a vector i
            i = image_stack[x, y, :]

            # construct diagonal matrix scriptI
            scripti = np.diag(i)

            # Multiply left (A) and right (B) sides by scriptI
            A = np.dot(scripti, scriptV)
            B = np.dot(scripti, i)

            # solve scriptI * scriptV * g = scriptI * i to obtain g for this point (usine least squares)
            g = np.linalg.lstsq(A, B, rcond=None)[0]

            # Calculate |g|
            norm_of_g = np.linalg.norm(g)

            # albedo at this point is |g|
            albedo[x, y] = norm_of_g
            # surface normal at this point is g / |g|
            normal[x, y, :] = g / norm_of_g

    return albedo, normal


if __name__ == '__main__':
    n = 5
    image_stack = np.zeros([10, 10, n])
    scriptV = np.zeros([n, 3])
    estimate_alb_nrm(image_stack, scriptV, shadow_trick=True)
