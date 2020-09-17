import numpy as np
import matplotlib.pyplot as plt
from math import pi


def createGabor(sigma, theta, lamda, psi, gamma):
    #CREATEGABOR Creates a complex valued Gabor filter.
    #   myGabor = createGabor( sigma, theta, lamda, psi, gamma ) generates
    #   Gabor kernels.
    #   - ARGUMENTS
    #     sigma      Standard deviation of Gaussian envelope.
    #     theta      Orientation of the Gaussian envelope. Takes arguments in
    #                the range [0, pi/2).
    #     lamda     The wavelength for the carriers. The central frequency
    #                (w_c) of the carrier signals.
    #     psi        Phase offset for the carrier signal, sin(w_c . t + psi).
    #     gamma      Controls the aspect ratio of the Gaussian envelope
    #
    #   - OUTPUT
    #     myGabor    A matrix of size [h,w,2], holding the real and imaginary
    #                parts of the Gabor in myGabor(:,:,1) and myGabor(:,:,2),
    #                respectively.

    # Set the aspect ratio.
    sigma_x = sigma
    sigma_y = sigma/gamma

    # Generate a grid
    nstds = 3
    xmax = max(abs(nstds*sigma_x*np.cos(theta)),
               abs(nstds*sigma_y*np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds*sigma_x*np.sin(theta)),
               abs(nstds*sigma_y*np.cos(theta)))
    ymax = np.ceil(max(1, ymax))

    # Make sure that we get square filters.
    xmax = max(xmax, ymax)
    ymax = max(xmax, ymax)
    xmin = -xmax
    ymin = -ymax

    # Generate a coordinate system in the range [xmin,xmax] and [ymin, ymax].
    [x, y] = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))

    # Convert to a 2-by-N matrix where N is the number of pixels in the kernel.
    XY = np.concatenate((x.reshape(1, -1), y.reshape(1, -1)), axis=0)

    # Compute the rotation of pixels by theta.
    # \\ Hint: Use the rotation matrix to compute the rotated pixel coordinates: rot(theta) * XY.
    rotMat = generateRotationMatrix(theta)
    rot_XY = np.matmul(rotMat, XY)
    # TODO I think this is wrong. Should be column-vectors, not row-vectors
    rot_x = rot_XY[0, :]
    rot_y = rot_XY[1, :]

    # Create the Gaussian envelope.
    # \\ IMPLEMENT the helper function createGauss.
    gaussianEnv = createGauss(rot_x, rot_y, gamma, sigma)

    # Create the orthogonal carrier signals.
    # \\ IMPLEMENT the helper functions createCos and createSin.
    cosCarrier = createCos(rot_x, lamda, psi)
    sinCarrier = createSin(rot_x, lamda, psi)

    # Modulate (multiply) Gaussian envelope with the carriers to compute
    # the real and imaginary components of the omplex Gabor filter.
    # modulate gaussianEnv with cosCarrier
    myGabor_real = gaussianEnv * cosCarrier
    # modulate gaussianEnv with sinCarrier
    myGabor_imaginary = gaussianEnv * sinCarrier

    # Pack myGabor_real and myGabor_imaginary into myGabor.
    h, w = myGabor_real.shape
    myGabor = np.zeros((h, w, 2))
    myGabor[:, :, 0] = myGabor_real
    myGabor[:, :, 1] = myGabor_imaginary

    # Uncomment below line to see how are the gabor filters
    # plt.imshow(myGabor_real)
    # plt.imshow(myGabor_imaginary)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(myGabor_real)    # Real
    # ax.axis("off")
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(myGabor_imaginary)    # Real
    # ax.axis("off")
    # plt.show()
    return myGabor


# Helper Functions
# ----------------------------------------------------------
def generateRotationMatrix(theta):
    # ----------------------------------------------------------
    # Returns the rotation matrix.
    # \\ Hint: https://en.wikipedia.org/wiki/Rotation_matrix \\
    # code the rotation matrix given theta.
    rotMat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    return rotMat

# ----------------------------------------------------------


def createCos(rot_x, lamda, psi):
    # ----------------------------------------------------------
    # Returns the 2D cosine carrier.
    # Implement the cosine given rot_x, lamda and psi.
    cosCarrier = np.cos(2*pi*rot_x / lamda+psi)

    # Reshape the vector representation to matrix.
    root_len = int(np.sqrt(cosCarrier.shape[0]))
    cosCarrier = np.reshape(cosCarrier, (root_len, -1))
    return cosCarrier

# ----------------------------------------------------------


def createSin(rot_x, lamda, psi):
    # ----------------------------------------------------------
    # Returns the 2D sine carrier.
    # Implement the sine given rot_x, lamda and psi.
    sinCarrier = np.sin(2*pi*rot_x / lamda+psi)

    # Reshape the vector representation to matrix.
    root_len = int(np.sqrt(sinCarrier.shape[0]))
    sinCarrier = np.reshape(sinCarrier, (root_len, -1))
    return sinCarrier

# ----------------------------------------------------------


def createGauss(rot_x, rot_y, gamma, sigma):
    # ----------------------------------------------------------
    # Returns the 2D Gaussian Envelope.
    # Implement the Gaussian envelope.
    gaussEnv = np.exp(-(rot_x**2 + gamma**2 * rot_y**2) / (2*sigma**2))
    # Reshape the vector representation to matrix.
    root_len = int(np.sqrt(gaussEnv.shape[0]))
    gaussEnv = np.reshape(gaussEnv, (root_len, -1))
    return gaussEnv


if __name__ == "__main__":

    fig, ax = plt.subplots(3, 5)

    cmap = 'gray'

    # Theta
    for i, val in enumerate([0, 1, 2, 3, 4]):
        real = createGabor(lamda=15, theta=pi*val/4, psi=0,
                           sigma=20, gamma=1)[..., 0]
        ax[0][i].imshow(real, cmap=cmap)
        ax[0][i].set_title(r"$\theta$="+str(val)+r"$\frac{\pi}{4}$")
        ax[0][i].axis('off')

    # Sigma
    for i, val in enumerate([3, 10, 20, 30, 50]):
        real = createGabor(lamda=15, theta=pi/2, psi=0,
                           sigma=val, gamma=1)[..., 0]
        ax[1][i].imshow(real, cmap=cmap)
        ax[1][i].set_title(r"$\sigma =$"+str(val))
        ax[1][i].axis('off')

    # Gamma
    for i, val in enumerate([0.1, 0.5, 1.0, 1.5, 10.0]):
        real = createGabor(lamda=15, theta=pi/2, psi=0,
                           sigma=20, gamma=val)[..., 0]
        ax[2][i].imshow(real, cmap=cmap)
        ax[2][i].set_title(r"$\gamma =$"+str(val))
        ax[2][i].axis('off')

    plt.tight_layout()
    plt.show()
