import numpy as np
import pyHMRF
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import groupby
from pre_process import model_selection


class CPT:
    def __init__(self, data_path, thin_layer_criteria=0.25, max_number_of_clusters=10, prior_mu=None, prior_mu_std=None,
                 prior_cov=None):
        """

        :param
        data_path: path to the dataset
        max_number_of_clusters: maximum possible number of clusters
        do_model_selection (bool): yes or no
        prior_mu: prior information of the center of each cluster, default is empty
        prior_mu_std: prior information of the std of the center of each cluster, default is empty
        prior_cov: prior information of the cov of each cluster, default is empty
        """

        # set the thin layer criteria
        self.thinLayerCriteria = thin_layer_criteria
        # initial data structure
        self.element = read_cpt_data(data_path)
        print('Number of data points:  ' + str(self.element.phys_shp))
        self.prior_mu = prior_mu
        self.prior_mu_std = prior_mu_std
        self.prior_cov = prior_cov
        if self.prior_mu is None:
            # perform model selection
            self.mod_sel = model_selection(self.element.feat, max_number_of_clusters, plot=True)
        # initialize layer_info with nan
        self.layer_info = np.nan

    def segmentation(self, num_of_iter, start_iter, beta_init=1, beta_jump_length=15):

        """
        :param
        :param num_of_iter: the number of iterations
        :param start_iter: the starting iter_ID of the converged Markov chain
        :param beta_init: initial value of beta
        :param beta_jump_length: the jump length of beta during MCMC sampling
        :return

        """

        if self.prior_mu is None:
            self.element.fit(n=num_of_iter, n_labels=self.mod_sel[1], beta_init=beta_init,
                             beta_jump_length=beta_jump_length)
        else:
            self.element.fit(n=num_of_iter, n_labels=len(self.prior_mu), beta_init=beta_init,
                             beta_jump_length=beta_jump_length,
                             prior_mu=self.prior_mu, prior_mu_std=self.prior_mu_std, prior_cov=self.prior_cov)
        self.element.get_estimator(start_iter=start_iter)
        self.element.get_label_prob(start_iter=start_iter)
        self.element.get_map()
        self.element.get_ie()

    def detect_layers(self):
        """

        :return:
            hardBoundary: location of hard layer boundaries
            softBoundary: location of soft layer boundaries
            n_layers: the number of layers
            prob_SBT: probability of each layer belonging to each SBT
            soil_SBT_type: SBT index of each layer
            label_of_layers: label of each layer

        """

        if type(self.element.label_map_est) is float:
            if np.isnan(self.element.label_map_est):
                print('Segmentation has not been performed!')
                return

        temp = np.concatenate(([0], np.diff(self.element.label_map_est)), axis=0)
        bnd_idx = temp != 0
        boundary_points = self.element.coords[bnd_idx] - 0.5 * (self.element.coords[2] - self.element.coords[1])
        thickness = np.diff(np.concatenate(([0], boundary_points.ravel(), self.element.coords[-1])))
        thin_idx = np.argwhere(thickness <= self.thinLayerCriteria)

        if len(thin_idx) > 0:
            for i in range(len(thin_idx)):
                if thin_idx[i] == 0:
                    boundary_points[0] = np.nan
                else:
                    if thin_idx[i] == len(boundary_points) + 1:
                        boundary_points[-1] = np.nan
                    else:
                        boundary_points[thin_idx[i] - 1] = 0.5 * (boundary_points[thin_idx[i] - 1] +
                                                                  boundary_points[thin_idx[i]])
                        boundary_points[thin_idx[i]] = np.nan

            boundary_points = np.array([boundary_points[~np.isnan(boundary_points)]]).T

        n_layers = len(boundary_points) + 1

        flag = np.concatenate(([0], boundary_points.ravel(), self.element.coords[-1] + 0.01))
        prob_sbt = np.zeros((9, n_layers))
        soil_sbt_type = np.zeros((n_layers,))
        label_of_layers = np.zeros((n_layers,))

        for i in range(n_layers):
            find_idx = (self.element.coords.ravel() > flag[i]) & (self.element.coords.ravel() < flag[i + 1])
            prob_sbt[:, i] = two_d_label(self.element.feat[find_idx, 0], self.element.feat[find_idx, 1])
            soil_sbt_type[i] = np.argmax(prob_sbt[:, i]) + 1
            temp = self.element.label_map_est[find_idx]
            unique, counts = np.unique(temp, return_counts=True)
            label_of_layers[i] = unique[np.argmax(counts)]

        hard_boundary = []
        soft_boundary = []

        for i in range(len(boundary_points)):
            if soil_sbt_type[i] != soil_sbt_type[i + 1]:
                hard_boundary.append(boundary_points[i])
            elif label_of_layers[i] != label_of_layers[i + 1]:
                soft_boundary.append(boundary_points[i])

        self.layer_info = [hard_boundary, soft_boundary, n_layers, prob_sbt, soil_sbt_type, label_of_layers]


def read_cpt_data(data_path):
    """    
    :param 
        data_path (str): path to the data file
        data file should contain 3 columns: [depth, Fr, Qt]
        data should start from the 1st row!
    :return: 
        
    """
    cpt_data = np.genfromtxt(data_path, delimiter=',')
    coord = np.array([np.nanmin(cpt_data[:, 0]), np.nanmax(cpt_data[:, 0])])
    data = np.log10(cpt_data[:, 1:])
    element = pyHMRF.Element(data, coord, normalize=False)
    return element


def two_d_label(log_fr, log_qt):
    temp = np.zeros([log_fr.shape[0], 2])
    temp[:, 0] = np.log(10 ** log_fr)
    temp[:, 1] = np.log(10 ** log_qt)
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


def mixture_plot_robertson_chart(cpt, res=50, figsize=(8, 4.8),
                                 cmap_contour='viridis', cmap_scatter='Pastel2', nlevels=3,
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

    n_label = cpt.element.mu_est.shape[0]

    if equal_axis:
        plt.axes().set_aspect('equal')
    else:
        plt.axes().set_aspect(aspect=aspect)

    for i in range(n_label):
        mu_temp = cpt.element.mu_est[i, :]
        cov_temp = cpt.element.cov_est[i, :, :]
        Z = multivariate_normal(mu_temp, cov_temp).pdf(pos)
        plt.contour(X, Y, Z, nlevels, cmap=cmap_contour, linewidths=lw)

    plt.scatter(cpt.element.feat[:, 0], cpt.element.feat[:, 1], s=5, c=cpt.element.label_map_est, cmap=cmap_scatter)

    # plot background Robertson chart
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
    plt.title('Statistical pattern on Robertson chart', fontsize=16)
    plt.xlabel(r'log$F_r$', fontsize=14)
    plt.ylabel(r'log$Q_t$', fontsize=14)
    plt.axis([-1, 1, 0, 3])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=4)

    plt.show()


def plot_layers(cpt, figsize=(12, 8), aspect=0.1):
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
    plt.plot(cpt.element.info_entr, cpt.element.coords, 'red', linewidth=1)
    plt.gca().invert_yaxis()  # y is increasing along depth
    plt.xlabel('Information entropy', fontsize=14)
    plt.ylabel('Depth (m)', fontsize=14)
    plt.title('Layer interpretation', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 1.25)

    # plot boundaries
    hard_boundary = np.tile(cpt.layer_info[0], 2)
    soft_boundary = np.tile(cpt.layer_info[1], 2)
    soil_sbt = cpt.layer_info[4]
    soil_sbt_unique = [x[0] for x in groupby(soil_sbt)]
    x_axis = np.array([0, 1.25])
    temp_1 = np.concatenate((cpt.element.coords[0], np.array(cpt.layer_info[0]).ravel()))
    temp_2 = np.concatenate((np.array(cpt.layer_info[0]).ravel(), cpt.element.coords[-1]))
    text_location_y = (temp_1+temp_2)/2
    text_location_x = 1

    if len(hard_boundary) > 0:
        for i in range(np.shape(hard_boundary)[0]):
            plt.plot(x_axis, hard_boundary[i, :], 'black', linestyle='-', linewidth=2)
    else:
        pass

    if len(soft_boundary) > 0:
        for i in range(np.shape(soft_boundary)[0]):
            plt.plot(x_axis, soft_boundary[i, :], 'grey', linestyle='--', linewidth=1)
    else:
        pass

    for i, value in enumerate(text_location_y):
        plt.text(text_location_x, value, 'SBT* = '+str(int(soil_sbt_unique[i])), fontsize=14)

    plt.grid(False)
    plt.show()
