import numpy as np
import pyHMRF
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import groupby


def read_cpt_data(datapath):
    """    
    :param 
        datapath (str): path to the data file 
    :return: 
        
    """
    cpt_data = np.genfromtxt(datapath, delimiter=',')
    coord = np.array([np.nanmin(cpt_data[:, 0]), np.nanmax(cpt_data[:, 0])])
    data = np.log10(cpt_data[:, 1:])
    cpt = pyHMRF.Element(data, coord, normalize=False)
    return cpt


def detect_layers(cpt, thinLayerCriteria=0.25):
    """

    :param
        cpt (:obj:`pyHMRF.Element`): the data structure for cpt data
        thinLayerCriteria (:obj: 'float'): thin layer crrteria

    :return:
        hardBoundary: location of hard layer boundaries
        softBoundary: location of soft layer boundaries
        n_layers: the number of layers
        prob_SBT: probability of each layer belonging to each SBT
        soil_SBT_type: SBT index of each layer
        label_of_layers: label of each layer

    """

    temp = np.concatenate(([0], np.diff(cpt.label_map_est)), axis=0)
    bnd_idx = temp != 0
    bndp = cpt.coords[bnd_idx] - 0.5 * (cpt.coords[2] - cpt.coords[1])
    thickness = np.diff(np.concatenate(([0], bndp.ravel(), cpt.coords[-1])))
    thin_idx = np.argwhere(thickness <= thinLayerCriteria)

    if len(thin_idx) > 0:
        for i in range(len(thin_idx)):
            if thin_idx[i] == 0:
                bndp[0] = np.nan
            else:
                if thin_idx[i] == len(bndp) + 1:
                    bndp[-1] = np.nan
                else:
                    bndp[thin_idx[i] - 1] = 0.5 * (bndp[thin_idx[i] - 1] + bndp[thin_idx[i]])
                    bndp[thin_idx[i]] = np.nan

        bndp = np.array([bndp[~np.isnan(bndp)]]).T

    n_layers = len(bndp) + 1

    flag = np.concatenate(([0], bndp.ravel(), cpt.coords[-1] + 0.01))
    prob_SBT = np.zeros((9, n_layers))
    soil_SBT_type = np.zeros((n_layers,))
    label_of_layers = np.zeros((n_layers,))

    for i in range(n_layers):
        find_idx = (cpt.coords.ravel() > flag[i]) & (cpt.coords.ravel() < flag[i + 1])
        prob_SBT[:, i] = TwoD_label(cpt.feat[find_idx, 0], cpt.feat[find_idx, 1])
        soil_SBT_type[i] = np.argmax(prob_SBT[:, i]) + 1
        temp = cpt.label_map_est[find_idx]
        unique, counts = np.unique(temp, return_counts=True)
        label_of_layers[i] = unique[np.argmax(counts)]

    hardBoundary = []
    softBoundary = []

    for i in range(len(bndp)):
        if soil_SBT_type[i] != soil_SBT_type[i + 1]:
            hardBoundary.append(bndp[i])
        elif label_of_layers[i] != label_of_layers[i + 1]:
            softBoundary.append(bndp[i])

    layer_info = [hardBoundary, softBoundary, n_layers, prob_SBT, soil_SBT_type, label_of_layers]
    return layer_info


def TwoD_label(log_Fr, log_Qt):
    temp = np.zeros([log_Fr.shape[0], 2])
    temp[:, 0] = np.log(10**log_Fr)
    temp[:, 1] = np.log(10**log_Qt)
    c = soil_label(temp)
    x = np.zeros([9, ])
    for i in range(x.shape[0]):
        x[i] = c[i].shape[0]/temp.shape[0]
    return x


def soil_label(x):
    """
    :param x:
    :return:
    purpose: soil classification based on Robertson (1990)
        The boundaries of SBT zones refer to Wang et al. (2013).
    """
    c_1 = x[-0.3707 * x[:, 0] ** 2 - 1.3625 * x[:, 0] + 1.0549 > x[:, 1]]
    x = np.delete(x, np.where(-0.3707 * x[:, 0] ** 2 - 1.3625 * x[:, 0] + 1.0549 > x[:, 1]), axis=0)
    temp = x[0.8095 * x[:, 0] ** 2 - 3.6795 * x[:, 0] + 8.1444 < x[:, 1]]
    x = np.delete(x, np.where(0.8095 * x[:, 0] ** 2 - 3.6795 * x[:, 0] + 8.1444 < x[:, 1]), axis=0)

    c_8 = temp[13.1077 * temp[:, 0] - 14.5023 < temp[:, 1]]
    c_9 = temp[13.1077 * temp[:, 0] - 14.5023 > temp[:, 1]]

    c_2 = x[(0.5586 * x[:, 0] ** 2 - 0.5399 * x[:, 0] + 0.3049 > x[:, 1]) & (x[:, 0] > 0.5)]
    x = np.delete(x, np.where((0.5586 * x[:, 0] ** 2 - 0.5399 * x[:, 0] + 0.3049 > x[:, 1]) & (x[:, 0] > 0.5)), axis=0)

    c_3 = x[(0.5405 * x[:, 0] ** 2 + 0.2739 * x[:, 0] + 1.6959 > x[:, 1]) & (x[:, 0] > -0.6)]
    x = np.delete(x, np.where((0.5405 * x[:, 0] ** 2 + 0.2739 * x[:, 0] + 1.6959 > x[:, 1]) & (x[:, 0] > -0.6)), axis=0)

    c_4 = x[(0.3833 * x[:, 0] ** 2 + 0.7805 * x[:, 0] + 2.5718 > x[:, 1]) & (x[:, 0] > -1.5)]
    x = np.delete(x, np.where((0.3833 * x[:, 0] ** 2 + 0.7805 * x[:, 0] + 2.5718 > x[:, 1]) & (x[:, 0] > -1.5)), axis=0)

    c_5 = x[0.2827 * x[:, 0] ** 2 + 0.967 * x[:, 0] + 4.1612 > x[:, 1]]
    x = np.delete(x, np.where(0.2827 * x[:, 0] ** 2 + 0.967 * x[:, 0] + 4.1612 > x[:, 1]), axis=0)

    c_6 = x[0.3477 * x[:, 0] ** 2 + 1.4933 * x[:, 0] + 6.6507 > x[:, 1]]
    x = np.delete(x, np.where(0.3477 * x[:, 0] ** 2 + 1.4933 * x[:, 0] + 6.6507 > x[:, 1]), axis=0)

    c_7 = x

    c = np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9])
    labeled_x = c

    return labeled_x


def mixture_plot_robertsonchart(cpt, res=50, figsize=(8, 4.8),
                                cmap_contour='viridis', cmap_scatter='Pastel2',
                                lw=1, equal_axis=False, aspect=0.8):
    """
    purpose: re-construct the SBT zones based on Robertson (1990)
        The boundaries of SBT zones refer to Wang et al. (2013).
    """
    plt.figure(figsize=figsize)
    # plot statistical pattern
    X = np.linspace(-1, 1, res)
    Y = np.linspace(0, 3, res)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    n_label = cpt.mu_est.shape[0]

    if equal_axis:
        plt.axes().set_aspect('equal')
    else:
        plt.axes().set_aspect(aspect=aspect)

    for i in range(n_label):
        mu_temp = cpt.mu_est[i, :]
        cov_temp = cpt.cov_est[i, :, :]
        Z = multivariate_normal(mu_temp, cov_temp).pdf(pos)
        plt.contour(X, Y, Z, cmap=cmap_contour, linewidths=lw)

    plt.scatter(cpt.feat[:, 0], cpt.feat[:, 1], s=5, c=cpt.label_map_est, cmap=cmap_scatter)

    # plot background Robertsonchart
    Fr = np.zeros([64, 8])
    Qt = np.zeros([64, 8])
    Fr[:, 0] = np.linspace(-2.3026, 0.6569, 64)
    Qt[:, 0] = -0.3707*Fr[:, 0]**2-1.3625*Fr[:, 0] + 1.0549

    Fr[:, 6] = np.linspace(0.3655, 2.3026, 64)
    Qt[:, 6] = 0.8095*Fr[:, 6]**2-3.6795*Fr[:, 6] + 8.1444

    Fr[:, 7] = np.linspace(1.4505, 1.6334, 64)
    Qt[:, 7] = 13.1077*Fr[:, 7] - 14.5023

    Fr[:, 1] = np.linspace(0.5589, 2.3026, 64)
    Qt[:, 1] = 0.5586*Fr[:, 1]**2-0.5399*Fr[:, 1] + 0.3049

    Fr[:, 2] = np.linspace(-0.5773, 1.8687, 64)
    Qt[:, 2] = 0.5405*Fr[:, 2]**2+0.2739*Fr[:, 2] + 1.6959

    Fr[:, 3] = np.linspace(-1.3334, 1.4505, 64)
    Qt[:, 3] = 0.3833*Fr[:, 3]**2+0.7805*Fr[:, 3] + 2.5718

    Fr[:, 4] = np.linspace(-2.3026, 0.9622, 64)
    Qt[:, 4] = 0.2827*Fr[:, 4]**2+0.967*Fr[:, 4] + 4.1612

    Fr[:, 5] = np.linspace(-2.3026, 0.1658, 64)
    Qt[:, 5] = 0.3477*Fr[:, 5]**2 + 1.4933*Fr[:, 5] + 6.6507

    Fr = np.log10(np.exp(Fr))
    Qt = np.log10(np.exp(Qt))

    for i in range(Fr.shape[1]):
        plt.plot(Fr[:, i], Qt[:, i], color='black', linewidth=1, zorder=-3)

    # setting axes parameters
    plt.title('Statistical pattern on Robertsonchart', fontsize=16)
    plt.xlabel(r'log$F_r$', fontsize=14)
    plt.ylabel(r'log$Q_t$', fontsize=14)
    plt.axis([-1, 1, 0, 3])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=4)

    plt.show()


def plot_layers(cpt, layer_info, figsize=(12, 8), aspect=0.1):
    """
    :param:
    cpt:
    bnd_info: output from detect_boundaries()
    figsize: figure size

    :return: plot information entropy and boundaries
    """

    # plot information entropy
    plt.figure(figsize=figsize)
    plt.axes().set_aspect(aspect=aspect)
    plt.plot(cpt.info_entr, cpt.coords, 'red', linewidth=1)  # rotate x axis to y axis,now y corresponds to depth
    plt.gca().invert_yaxis()  # y is increasing along depth
    plt.xlabel('Information entropy', fontsize=14)
    plt.ylabel('Depth (m)', fontsize=14)
    plt.title('Layer interpretation', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 1.25)

    # plot boundaries
    hardboundary = np.tile(layer_info[0], 2)
    softboundary = np.tile(layer_info[1], 2)
    soil_SBT = layer_info[4]
    soil_SBT_unique = [x[0] for x in groupby(soil_SBT)]
    xaxis = np.array([0, 1.25])
    temp_1 = np.concatenate((cpt.coords[0], np.array(layer_info[0]).ravel()))
    temp_2 = np.concatenate((np.array(layer_info[0]).ravel(), cpt.coords[-1]))
    text_location_y = (temp_1+temp_2)/2
    text_location_x = 1

    if len(hardboundary) > 0:
        for i in range(np.shape(hardboundary)[0]):
            plt.plot(xaxis, hardboundary[i, :], 'black', linestyle='-', linewidth=2)
    else:
        pass

    if len(softboundary) > 0:
        for i in range(np.shape(softboundary)[0]):
            plt.plot(xaxis, softboundary[i, :], 'grey', linestyle='--', linewidth=1)
    else:
        pass

    for i, value in enumerate(text_location_y):
        plt.text(text_location_x, value, 'SBT* = '+str(int(soil_SBT_unique[i])), fontsize=14)

    plt.grid(False)
    plt.show()


def segmentation(cpt, n, n_labels, start_iter, beta_init=1, beta_jump_length=15):
    """

    :param cpt: the object from pyHMRF.Element()
    :param n: the number of iterations
    :param n_labels: the number of labels
    :param start_iter: the starting iter_ID of the converged Markov chain
    :param beta_init: initial value of beta
    :param beta_jump_length: the jump length of beta during MCMC sampling
    :return:
    """

    cpt.fit(n, n_labels, beta_init, beta_jump_length)
    cpt.get_estimator(start_iter=start_iter)
    cpt.get_label_prob(start_iter=start_iter)
    cpt.get_map()
    cpt.get_ie()