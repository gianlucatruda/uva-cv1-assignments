import cv2
import matplotlib.pyplot as plt
import math
from math import pi
import numpy as np
import time
import sklearn

from createGabor import createGabor, createGauss


def main(k=2, smoothing_flag=True, standard=True, image='Kobi',
         gauss_sigma=None,
         lambdas=None,
         gabor_sigmas=None,
         thetas=None,
         vis_mode=False):

    # Hyperparameters
    # k = 2      # number of clusters in k-means algorithm. By default,
    # we consider k to be 2 in foreground-background segmentation task.

    image_id = image  # Identifier to switch between input images.
    # Possible ids: 'Kobi',    'Polar', 'Robin-1'
    #               'Robin-2', 'Cows', 'SciencePark'

    # Misc
    err_msg = 'Image not available.'

    # Control settings
    visFlag = vis_mode  # Set to true to visualize filter responses.
    # Set to true to postprocess filter outputs.
    smoothingFlag = smoothing_flag

    # Read image
    if image_id == 'Kobi':
        img = cv2.imread('./data/kobi2.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.25
    elif image_id == 'Polar':
        img = cv2.imread('./data/polar-bear-hiding.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.75
    elif image_id == 'Robin-1':
        img = cv2.imread('./data/robin-1.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 1
    elif image_id == 'Robin-2':
        img = cv2.imread('./data/robin-2.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.5
    elif image_id == 'Cows':
        img = cv2.imread('./data/cows.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.5
    elif image_id == 'SciencePark':
        img = cv2.imread('./data/sciencepark.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_factor = 0.2
    else:
        raise ValueError(err_msg)

    # Image adjustments
    img_raw = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)

    # Display image
    if standard:
        plt.figure()
        plt.title(f'Input image: {image_id}')
        plt.imshow(img_raw, cmap='gray')
        plt.axis("off")
        plt.show()

    # Design array of Gabor Filters
    # In this code section, you will create a Gabor Filterbank. A filterbank is
    # a collection of filters with varying properties (e.g. {shape, texture}).
    # A Gabor filterbank consists of Gabor filters of distinct orientations
    # and scales. We will use this bank to extract texture information from the
    # input image.

    numRows, numCols = img.shape

    # Estimate the minimum and maximum of the wavelengths for the sinusoidal
    # carriers.
    # ** This step is pretty much standard, therefore, you don't have to
    #    worry about it. It is cycles in pixels. **
    lambdaMin = 4/np.sqrt(2)
    lambdaMax = np.sqrt(abs(numRows)**2 + abs(numCols)**2)

    # Specify the carrier wavelengths.
    # (or the central frequency of the carrier signal, which is 1/lambda)
    n = np.floor(np.log2(lambdaMax/lambdaMin))
    if lambdas is None:
        lambdas = 2**np.arange(0, (n-2)+1) * lambdaMin

    # Define the set of orientations for the Gaussian envelope.
    dTheta = 2 * np.pi/8                  # \\ the step size
    if thetas is None:
        orientations = np.arange(0, np.pi+dTheta, dTheta)
    else:
        orientations = thetas

    # Define the set of sigmas for the Gaussian envelope. Sigma here defines
    # the standard deviation, or the spread of the Gaussian.

    if gabor_sigmas is None:
        sigmas = np.array([1, 2])
    else:
        sigmas = np.array(gabor_sigmas)

    if gauss_sigma is None:
        gauss_sigma = 5

    # Notify user what the configurations are
    print(f"Lambdas: {lambdas}")
    print(f"Gabor sigmas: {sigmas}")
    print(f"Gauss sigma: {gauss_sigma}")
    print(f"Thetas: {orientations}")

    # Now you can create the filterbank. We provide you with a Python list
    # called gaborFilterBank in which we will hold the filters and their
    # corresponding parameters such as sigma, lambda and etc.
    # ** All you need to do is to implement createGabor(). Rest will be handled
    #    by the provided code block. **
    gaborFilterBank = []
    tic = time.time()
    for lmbda in lambdas:
        for sigma in sigmas:
            for theta in orientations:
                # Filter parameter configuration for this filter.
                psi = 0
                gamma = 0.5

                # Create a Gabor filter with the specs above.
                # (We also record the settings in which they are created. )
                # // TODO: Implement the function createGabor() following
                #          the guidelines in the given function template.
                #          ** See createGabor.py for instructions ** //
                filter_config = {}
                filter_config["filterPairs"] = createGabor(
                    sigma, theta, lmbda, psi, gamma)
                filter_config["sigma"] = sigma
                filter_config["lmbda"] = lmbda
                filter_config["theta"] = theta
                filter_config["psi"] = psi
                filter_config["gamma"] = gamma
                gaborFilterBank.append(filter_config)
    ctime = time.time() - tic

    print('--------------------------------------\n \t\tDetails\n--------------------------------------')
    print(f'Total number of filters       : {len(gaborFilterBank)}')
    print(f'Number of scales (sigma)      : {len(sigmas)}')
    print(f'Number of orientations (theta): {len(orientations)}')
    print(f'Number of carriers (lambda)   : {len(lambdas)}')
    print(f'---------------------------------------')
    print(f'Filter bank created in {ctime} seconds.')
    print(f'---------------------------------------')

    # Filter images using Gabor filter bank using quadrature pairs (real and imaginary parts)
    # You will now filter the input image with each complex Gabor filter in
    # gaborFilterBank structure and store the output in the cell called
    # featureMaps.
    # // Hint-1: Apply both the real imaginary parts of each kernel
    #            separately in the spatial domain (i.e. over the image). //
    # // Hint-2: Assign each output (i.e. real and imaginary parts) in
    #            variables called real_out and imag_out. //
    # // Hint-3: Use built-in cv2 function, filter2D, to convolve the filter
    #            with the input image. Check the options for padding. Find
    #            the one that works well. You might want to
    #            explain what works better and why shortly in the report.
    featureMaps = []

    for gaborFilter in gaborFilterBank:
        # gaborFilter["filterPairs"] has two elements. One is related to the real part
        # of the Gabor Filter and the other one is the imagineray part.

        # filter the grayscale input with real part of the Gabor
        real_out = cv2.filter2D(img, -1, gaborFilter['filterPairs'][:, :, 0])
        # filter the grayscale input with imaginary part of the Gabor
        imag_out = cv2.filter2D(img, -1, gaborFilter['filterPairs'][:, :, 1])

        featureMaps.append(np.stack((real_out, imag_out), 2))

        # Visualize the filter responses if you wish.
        # if visFlag:
        #     fig = plt.figure()

        #     ax = fig.add_subplot(1, 2, 1)
        #     ax.imshow(real_out)    # Real
        #     title = "Re[h(x,y)], \n lambda = {0:.4f}, \n theta = {1:.4f}, \n sigma = {2:.4f}".format(
        #         gaborFilter["lmbda"], gaborFilter["theta"], gaborFilter["sigma"])
        #     ax.set_title(title)
        #     ax.axis("off")

        #     ax = fig.add_subplot(1, 2, 2)
        #     ax.imshow(imag_out)    # Real
        #     title = "Im[h(x,y)], \n lambda = {0:.4f}, \n theta = {1:.4f}, \n sigma = {2:.4f}".format(
        #         gaborFilter["lmbda"], gaborFilter["theta"], gaborFilter["sigma"])
        #     ax.set_title(title)
        #     ax.axis("off")
        #     plt.show()

    # Compute the magnitude
    # Now, you will compute the magnitude of the output responses.
    # \\ Hint: (real_part^2 + imaginary_part^2)^(1/2) \\
    featureMags = []
    for i, fm in enumerate(featureMaps):
        real_part = fm[..., 0]
        imag_part = fm[..., 1]
        mag = np.sqrt(real_part**2 + imag_part**2)
        featureMags.append(mag)

        # Visualize the magnitude response if you wish.
        if visFlag:
            fig = plt.figure()

            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mag.astype(np.uint8))    # visualize magnitude
            title = "Re[h(x,y)], \n lambda = {0:.4f}, \n theta = {1:.4f}, \n sigma = {2:.4f}".format(
                gaborFilterBank[i]["lmbda"],
                gaborFilterBank[i]["theta"],
                gaborFilterBank[i]["sigma"])
            ax.set_title(title)
            ax.axis("off")

    # Prepare and Preprocess features
    # You can think of each filter response as a sort of feature representation
    # for the pixels. Now that you have numFilters = |gaborFilterBank| filters,
    # we can represent each pixel by this many features.
    # \\ Q: What kind of features do you think gabor filters might correspond to?

    # You will now implement a smoothing operation over the magnitude images in
    # featureMags.
    # \\ Hint: For each i in [1, length(featureMags)], smooth featureMags{i}
    #          using an appropriate first order Gaussian kernel.
    # \\ Hint: cv2 filter2D function is helpful here.
    features = np.zeros(shape=(numRows, numCols, len(featureMags)))
    if smoothingFlag:

        for i, fmag in enumerate(featureMags):
            # i)  filter the magnitude response with appropriate Gaussian kernels
            gauss_vec = cv2.getGaussianKernel(
                int(2*np.ceil(2*5)+1), gauss_sigma)
            gauss_kernel = gauss_vec * gauss_vec.T
            # ii) insert the smoothed image into features[:,:,jj]
            features[:, :, i] = cv2.filter2D(
                featureMags[i].astype('float64'), -1, gauss_kernel)
    else:
        # Don't smooth but just insert magnitude images into the matrix
        # called features.
        for i, fmag in enumerate(featureMags):
            features[:, :, i] = fmag

    # Reshape the filter outputs (i.e. tensor called features) of size
    # [numRows, numCols, numFilters] into a matrix of size [numRows*numCols, numFilters]
    # This will constitute our data matrix which represents each pixel in the
    # input image with numFilters features.
    features = np.reshape(features, newshape=(numRows * numCols, -1))

    # Standardize features.
    # \\ Hint: see http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing for more information.

    #  i)  Implement standardization on matrix called features.
    #  ii) Return the standardized data matrix.

    features = (features - np.mean(features)) / np.std(features)

    # (Optional) Visualize the saliency map using the first principal component
    # of the features matrix. It will be useful to diagnose possible problems
    # with the pipeline and filterbank.
    from sklearn.decomposition import PCA
    transformed_feature = PCA(n_components=1).fit_transform(
        features)  # select the first component
    transformed_feature = np.ascontiguousarray(
        transformed_feature, dtype=np.float32)
    feature2DImage = np.reshape(
        transformed_feature, newshape=(numRows, numCols))
    if standard:
        plt.figure()
        plt.title(f'Pixel representation projected onto first PC')
        plt.imshow(feature2DImage, cmap='gray')
        plt.axis("off")
        plt.show()

    # Apply k-means algorithm to cluster pixels using the data matrix,
    # features.
    # \\ Hint-1: search about sklearn kmeans function https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.
    # \\ Hint-2: use the parameter k defined in the first section when calling
    #            sklearn's built-in kmeans function.
    tic = time.time()

    from sklearn.cluster import KMeans
    kmeans_trans = KMeans(n_clusters=k)

    # Return cluster labels per pixel
    pixLabels = kmeans_trans.fit_predict(features)

    ctime = time.time() - tic
    print(f'Clustering completed in {ctime} seconds.')

    # Visualize the clustering by reshaping pixLabels into original grayscale
    # input size [numRows numCols].
    pixLabels = np.reshape(pixLabels, newshape=(numRows, numCols))
    if standard:
        plt.figure()
        plt.title(f'Pixel clusters')
        plt.imshow(pixLabels)
        plt.axis("off")
        plt.show()

    # Use the pixLabels to visualize segmentation.
    Aseg1 = np.zeros_like(img_raw)
    Aseg2 = np.zeros_like(img_raw)
    BW = pixLabels == 1
    BW = np.repeat(BW[:, :, np.newaxis], 3, axis=2)
    Aseg1[BW] = img_raw[BW]
    Aseg2[~BW] = img_raw[~BW]

    if standard:
        plt.figure()
        plt.title(f'montage')
        plt.imshow(Aseg1, 'gray', interpolation='none')
        plt.imshow(Aseg2, 'jet',  interpolation='none', alpha=0.7)
        plt.axis("off")
        plt.show()

    if not standard:
        return img_raw, img, Aseg1, Aseg2


if __name__ == '__main__':
    # Broke: Run the code with all the bugs (as provided)

    # Woke: Run the code fixed up so it actually works
    # main()

    # Bespoke: Run the code in a modular way on all images and make a useful multiplot

    images = ['Kobi', 'Polar', 'Robin-1', 'Robin-2', 'Cows', 'SciencePark']

    hyperparams = {
        'Kobi': {
            'lambdas': [5.65685425], 'thetas': None, 'gabor_sigmas': [0.5, 0.75, 1.0, 2.0, 3.0], 'gauss_sigma': 25,
        },
        'Polar': {
            'lambdas': [2.82842712, 11.3137085, 22.627417], 'thetas': None, 'gabor_sigmas': [0.75, 1.0], 'gauss_sigma': 25,
        },
        'Robin-1': {
            'lambdas': [2.82842712, 11.3137085, 22.627417], 'thetas': None, 'gabor_sigmas': [0.75, 1.0], 'gauss_sigma': 3,
        },
        'Robin-2': {
            'lambdas': [2.82842712, 22.627417, 90.50966799], 'thetas': None, 'gabor_sigmas': [0.5, 0.75, 1.0], 'gauss_sigma': 20,
        },
        'Cows': {
            'lambdas': [2.82842712, 5.65685425], 'thetas': None, 'gabor_sigmas': [0.25, 0.5, 0.75, 1.5, 2.0, ], 'gauss_sigma': 6,
        },
        'SciencePark': {
            'lambdas': None, 'thetas': [0.78539816, 1.57079633, 2.35619449], 'gabor_sigmas': [2.0], 'gauss_sigma': 15,
        }
    }

    fig, ax = plt.subplots(len(images), 3, figsize=(5, 7))
    for i, img in enumerate(images):
        img_raw, img, Aseg1, Aseg2 = main(
            k=2,
            smoothing_flag=True,
            standard=False,
            image=img,
            **hyperparams[img],
        )
        ax[i][0].imshow(img_raw)
        ax[i][0].axis("off")
        ax[i][1].imshow(Aseg1, 'gray', interpolation='none')
        ax[i][1].axis("off")
        ax[i][2].imshow(Aseg2, 'gray',  interpolation='none')
        ax[i][2].axis("off")
    plt.tight_layout()
    plt.show()
