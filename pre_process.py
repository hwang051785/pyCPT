# import gdal
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture  # gaussian mixture model
import itertools
from scipy import linalg
from matplotlib import patches


# def extract_data(data_path):
#     # open tif file
#     src_ds = gdal.Open(data_path)  # src_ds: source data-set
#
#     # get data matrix
#     print("[ RASTER BAND COUNT ]: ", src_ds.RasterCount)
#
#     data = np.array([])
#     for band in range(src_ds.RasterCount):
#         band += 1
#         print("[ GETTING BAND ]: ", band)
#         src_band = src_ds.GetRasterBand(band)
#         temp = src_band.ReadAsArray()
#         if not data:
#             data = temp
#         else:
#             data = np.stack((data, temp), axis=2)
#
#     # get corner coordinate matrix
#     src_ds.GetProjectionRef()  # get coordinate info
#     width = src_ds.RasterXSize
#     height = src_ds.RasterYSize
#     gt = src_ds.GetGeoTransform()
#     min_x = gt[0]
#     min_y = gt[3] + width * gt[4] + height * gt[5]
#     max_x = gt[0] + width * gt[1] + height * gt[2]
#     max_y = gt[3]
#     coord = np.array([[min_x, max_x], [min_y, max_y]])
#
#     return data, coord


def model_selection(feat, n_labels, tol=5e-5, reg_covar=1e-3, max_iter=1000, n_init=100, plot=False):
    """Plots the Bayesian Information Criterion of Gaussian Mixture Models for the given features and range of labels
    defined by the given upper boundary.

    Args:
        feat (:obj:`np.ndarray`): Feature vector containing the data in a flattened format.
        n_labels (int): Sets the included upper bound for the number of features to be considered in the analysis.
        tol (float): tolerance of difference at converge
        reg_covar (float): regularization value of covariance matrix
        max_iter (int): max number of iteration
        n_init (int): number of initial trials
        plot (bool): plot bic

    Returns:
        bic_list
        Plot (if plot == True)

    """
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, n_labels+1)
    cv_types = ['full']
    best_gmm = np.nan
    for cv_type in cv_types:
        for n_components in n_components_range:
            print('Fitting model with number of components = ', n_components)
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type,
                                          tol=tol,
                                          reg_covar=reg_covar,
                                          max_iter=max_iter,
                                          n_init=n_init)
            gmm.fit(feat)
            bic.append(gmm.bic(feat))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    if plot:
        spl = plt.subplot(2, 1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))

        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC score per model')
        xpos = np.mod(bic.argmin(), len(n_components_range)) + 0.65 + \
            0.2 * np.floor(bic.argmin() / len(n_components_range))
        plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], cv_types)
        # Plot the winner
        splot = plt.subplot(2, 1, 2)
        Y_ = clf.predict(feat)

        color_box = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

        for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                                   color_box)):
            v, w = linalg.eigh(cov)
            if not np.any(Y_ == i):
                continue
            plt.scatter(feat[Y_ == i, 0], feat[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180. * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(.5)
            splot.add_artist(ell)

        plt.xticks(())
        plt.yticks(())
        plt.title('Selected GMM: full model')
        plt.subplots_adjust(hspace=.7, bottom=.02)
        plt.show()

    return bic, bic.argmin() + 1

