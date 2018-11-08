import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import gridspec, rcParams  # plot arrangements
import math
import scipy.io

plt.style.use('bmh')  # plot style


def plot_statistical_pattern(element, res=50, xlabel='Feature_1', ylabel='Feature_2', figsize=(10, 6),
                             cmap_contour='viridis', cmap_scatter='Pastel2', lw=1, equal_axis=False, aspect=0.5):
    X = np.linspace(min(element.feat[:, 0]), max(element.feat[:, 0]), res)
    Y = np.linspace(min(element.feat[:, 1]), max(element.feat[:, 1]), res)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    n_label = element.mu_est.shape[0]
    plt.figure(figsize=figsize)
    if equal_axis:
        plt.axes().set_aspect('equal')
    else:
        plt.axes().set_aspect(aspect=aspect)

    for i in range(n_label):
        mu_temp = element.mu_est[i, :]
        cov_temp = element.cov_est[i, :, :]
        Z = multivariate_normal(mu_temp, cov_temp).pdf(pos)
        plt.contour(X, Y, Z, cmap=cmap_contour, linewidths=lw)

    plt.scatter(element.feat[:, 0], element.feat[:, 1], s=5, c=element.label_map_est, cmap=cmap_scatter)
    plt.title('Gaussian mixture plot')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)

    plt.show()


def plot_image(element, list_field, title='', xlabel='', ylabel='', figsize=(12, 8), cmap='tab20', colorbar=True, scatter=True):
    """Diagnostic plots for plotting a specific field from the 1-D list structure.
      Args:
          element: an object of Class Element
          list_field : 1-D list of a specific field
          title : title of plot
          xlabel: xlabel
          ylabel: ylabel
          colorbar: plot colorbar
          scatter: plot style for plotting 1-Dimensional phy_dim
          figsize: set figsize
          cmap: color map
      Returns:
          Plot
    """

    # plot 1D data
    if element.phyDim == 1:
        plt.figure(figsize=figsize)
        rcParams.update({'font.size': 8})
        plt.title(title)
        field = np.array(list_field)
        if scatter:
            plt.scatter(field, element.coords, s=40,
                        c=list_field)  # rotate x axis to y axis,now y corresponds to depth
            plt.gca().invert_yaxis()  # y is increasing along depth
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        else:
            plt.plot(field, element.coords, linewidth=1)  # rotate x axis to y axis,now y corresponds to depth
            plt.gca().invert_yaxis()  # y is increasing along depth
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

    # plot 2D data
    elif element.phyDim == 2:
        plt.figure(figsize=figsize)
        rcParams.update({'font.size': 8})

        field = np.asmatrix(np.array(list_field).reshape(element.phys_shp[0], element.phys_shp[1]))
        image = plt.imshow(field, cmap=cmap)
        plt.title(title)
        plt.grid(False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if colorbar:
            plt.colorbar(image, fraction=0.046, pad=0.04)
        else:
            pass

    # plot 3D data
    elif element.phyDim == 3:
        # 3d case
        raise Exception("3D segmentation not yet supported.")

        # mismatch
    else:
        raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")
    plt.show()


def plot_feature(element, xlabel='', ylabel='', figsize=(12, 8), scatter=True):
    gs = gridspec.GridSpec(math.ceil(element.n_feat / 2), 2)
    rcParams.update({'font.size': 8})
    plt.figure(figsize=figsize)
    # plot 1D data
    if element.phyDim == 1:
        if scatter:
            for i in range(element.n_feat):
                i = i + 1
                if i % 2 == 0:
                    ax = plt.subplot(gs[math.ceil(i / 2) - 1, 1])
                else:
                    ax = plt.subplot(gs[math.ceil(i / 2) - 1, 0])
                ax.set_title('Feature_' + str(i))
                ax.scatter(element.feat[:, i-1], element.coords, s=40,
                                   c=element.label_map_est)  # rotate x axis to y axis,now y corresponds to depth
                plt.xlabel(xlabel[i-1])
                plt.gca().invert_yaxis()  # y is increasing along depth
                plt.ylabel(ylabel)
        else:
            for i in range(element.n_feat):
                i = i + 1
                if i % 2 == 0:
                    ax = plt.subplot(gs[math.ceil(i / 2) - 1, 1])
                else:
                    ax = plt.subplot(gs[math.ceil(i / 2) - 1, 0])
                ax.set_title('Feature_' + str(i))
                ax.plot(element.feat[:, i-1], element.coords, linewidth=1)  # rotate x axis to y axis,now y corresponds to depth
                plt.xlabel(xlabel[i-1])
                plt.gca().invert_yaxis()  # y is increasing along depth
                plt.ylabel(ylabel)

    # plot 2D data
    elif element.phyDim == 2:
        for i in range(element.n_feat):
            i = i + 1
            if i % 2 == 0:
                ax = plt.subplot(gs[math.ceil(i / 2) - 1, 1])
            else:
                ax = plt.subplot(gs[math.ceil(i / 2) - 1, 0])
            ax.set_title('Feature_' + str(i))
            field = np.asmatrix(np.array(element.feat[:, i-1]).reshape(element.phys_shp[0], element.phys_shp[1]))
            image = ax.imshow(field, cmap="viridis")
            plt.colorbar(image, fraction=0.046, pad=0.04)
            plt.grid(False)
        plt.xlabel(xlabel)
        plt.ylabel(xlabel)

    # plot 3D data
    elif element.phyDim == 3:
        # 3d case
        raise Exception("3D segmentation not yet supported.")
    plt.show()


def write_mat(file_name, element):
    mdict = {
        'cpt_mu_est': element.mu_est,
        'cpt_cov_est': element.cov_est.transpose((1, 2, 0)),
        'cpt_label_map_est': np.asmatrix(element.label_map_est).T,
        'cpt_label_prob': np.asmatrix(element.label_prob).T,
        'cpt_label_bin': np.asmatrix(element.labels).T,
        'cpt_entropy': np.asmatrix(element.info_entr).T
    }
    scipy.io.savemat(file_name, mdict)
