import numpy as np


def estimate_alb_nrm(image_stack, scriptV, shadow_trick=False, colour=False):

    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # colour: (true/false) whether the image is in colour (extra dimension for channels)
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    if shadow_trick:
        print("Using shadow trick!\n")

    # Expand dimensions by 1 if grayscale
    if not colour:
        image_stack = np.expand_dims(image_stack, 3)
        scriptV = np.expand_dims(scriptV, 2)

    h, w, n, c = image_stack.shape

    # create arrays for
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w, 3])
    normal = np.zeros([h, w, 3, 3])

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

    # Loop through channels
    for ch in range(c):
        # for each point in image_stack array
        for x in range(h):
            for y in range(w):

                # stack image values for channel into a vector i
                i = image_stack[x, y, :, ch]
                # Get scriptV for current channel
                _scriptV = scriptV[:, :, ch]

                if shadow_trick:
                    # construct diagonal matrix scriptI
                    scripti = np.diag(i)
                    # Multiply left (A) and right (B) sides by scriptI
                    A = np.dot(scripti, _scriptV)
                    B = np.dot(scripti, i)
                else:
                    # Use the standard `i = V . g` formula
                    A = _scriptV
                    B = i

                # solve scriptI * scriptV * g = scriptI * i to obtain g for this point (usine least squares)
                g = np.linalg.lstsq(A, B, rcond=None)[0]

                # Calculate |g|
                norm_of_g = np.linalg.norm(g)

                # albedo at this point is |g|
                albedo[x, y, ch] = norm_of_g

                # surface normal at this point is g / |g|
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Ignore divide by zero errors
                    normal[x, y, :, ch] = np.divide(g, norm_of_g)

    # Take mean of the normal across channels
    if colour:
        # Ignore nans if colour. Only undefined if nan for ALL channels.
        normal = np.nanmean(normal, axis=3)
    else:
        normal = np.mean(normal, axis=3)

    # Collapse albedo colour channels if grayscale
    if not colour:
        albedo = np.nanmean(albedo, axis=2)

    return albedo, normal


if __name__ == '__main__':
    n = 5
    image_stack = np.zeros([10, 10, n])
    scriptV = np.zeros([n, 3])
    estimate_alb_nrm(image_stack, scriptV, shadow_trick=False)