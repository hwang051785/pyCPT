"""
pyHMRF is a vectorized Python library for unsupervised clustering of data sets with n-dimensional (n>=2) features,
designed for the segmentation of one-, two- and three-dimensional data in the field of remote sensing, geological
modeling and geophysics.The library is based on the algorithm developed by Wang et al., 2017 and combines Hidden Markov
Random Fields with Gaussian Mixture Models in a Bayesian inference framework.

************************************************************************************************
References

[1] Wang, H., Wellmann, J. F., Li, Z., Wang, X., & Liang, R. Y. (2017). A Segmentation Approach for Stochastic
    Geological Modeling Using Hidden Markov Random Fields. Mathematical Geosciences, 49(2), 145-177.

************************************************************************************************
@authors: Tianqi Zhang, Hui Wang, Alexander Schaaf
************************************************************************************************
pyHMRF is licensed under the GNU Lesser General Public License v3.0
************************************************************************************************
"""

import numpy as np  # scientific computing library
import math
from sklearn import mixture  # gaussian mixture model
from scipy.stats import multivariate_normal, norm  # normal distributions
from copy import copy
from itertools import combinations
import tqdm  # smart-ish progress bar


class Element:
    def __init__(self, data, coord, stencil=None, standardize=False, normalize=True):
        """
        Args:
            data (:obj:`np.ndarray`): Multidimensional data array containing all observations (features) in the
                following format:

                    1D scenario: data = F[coord_idx, feature_idx];
                    2D scenario: data = F[coord_x_idx, coord_y_idx, feature_idx];
                    3D scenario: data = F[coord_x_idx, coord_y_idx, coord_z_idx, feature_idx]

            coord (:obj:'np.array'): two-dimensional matrix containing the first and last coordinate of each physical
                                     dimension
                following format:

                    1D scenario: coord = [y_0, y_n]
                    2D scenario: coord = [[y_0, y_n],
                                          [x_0, x_n]]
                    3D scenario: coord = [[y_0, y_n],
                                          [x_0, x_n],
                                          [z_0, z_n]]
                    Note: x,y,z are coordinates

            stencil (int): Number specifying the stencil of the neighborhood system used in the Gibbs energy
                calculation.
            standardize (bool): standardize the data
            normalize (bool): normalize the data
        """
        # TODO: [DOCS] Main object description

        # store initial data
        self.data = data
        # get number of rows and columns of the database
        self.shape = np.shape(data)
        # store physical dimension
        self.phyDim = np.shape(np.asmatrix(coord))[0]
        if self.phyDim == 1:
            self.phys_shp = np.array(self.shape[0:1])
        elif self.phyDim == 2:
            self.phys_shp = np.array(self.shape[0:2])
        elif self.phyDim == 3:
            self.phys_shp = np.array(self.shape[0:3])
        else:
            raise Exception("Physical space is up to 3-D!")

        # get number of features
        if len(self.shape) > self.phyDim:
            self.n_feat = self.shape[-1]
        else:
            self.n_feat = 1

        # calculate the total number of pixels
        self.num_pixels = np.prod(self.phys_shp)

        # implement the pseudo-color method
        self.stencil = stencil
        self.colors = pseudocolor(self.phys_shp, self.stencil)

        # ************************************************************************************************
        # fetch dimensionality, coordinate and feature vector from input data
        # feature matrix/vector has the shape: num_pixel by n_feat

        # 1D scenario
        if self.phyDim == 1:
            # calculate the coordinate increment
            # y_0 = coord[0]; y_n = coord[1]
            delta_coord = (coord[1] - coord[0])/(self.shape[0] - 1)
            # create coordinate vector
            self.coords = np.array([np.arange(coord[0], coord[1]+delta_coord, delta_coord)]).T  # an n-by-one matrix
            # create feature vector
            self.feat = self.data

        # 2D scenario
        elif self.phyDim == 2:
            # calculate the coordinate increment
            # x_0 = coord[1, 0];    x_n = coord[1, 1];
            # y_0 = coord[0, 0];    y_n = coord[0, 1];
            delta_coord_y = (coord[0, 1] - coord[0, 0]) / (self.shape[0] - 1)
            delta_coord_x = (coord[1, 1] - coord[1, 0]) / (self.shape[1] - 1)
            # create coordinate vector
            y, x = np.indices(self.shape[0:2])
            self.coords = np.array([y.flatten(), x.flatten()]).T
            self.coords[:, 0] = (self.coords[:, 0].max() - self.coords[:, 0]) * delta_coord_y + coord[0, 0]
            self.coords[:, 1] = self.coords[:, 1] * delta_coord_x + coord[1, 0]
            # create feature matrix
            if len(self.shape) == 2:
                self.feat = np.array([self.data.ravel()]).T
            else:
                self.feat = np.array([self.data[:, :, f].ravel() for f in range(self.n_feat)]).T

        # 3D scenario
        elif self.phyDim == 3:
            raise Exception("3D segmentation not yet supported.")

        # physical dimension mismatch
        else:
            raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")

        if standardize:
            self.standardize_feature_vectors()

        if normalize:
            self.normalize_feature_vectors()

        self.n_labels = np.nan
        self.gmm = np.nan
        self.labels = []
        self.mus = []
        self.covs = []
        self.betas = []
        # Initiate the variables
        self.storage_te = []
        self.beta_acc_ratio = np.array([])
        self.cov_acc_ratio = np.array([])
        self.mu_acc_ratio = np.array([])
        # initialize estimators with nan values
        self.mu_est = np.nan
        self.mu_std = np.nan
        self.cov_est = np.nan
        self.cov_std = np.nan
        self.beta_est = np.nan
        self.beta_std = np.nan
        self.label_prob = np.nan
        self.label_map_est = np.nan
        self.info_entr = np.nan
        self.beta_dim = np.nan
        # initialize the priors with nan
        self.prior_beta = np.nan
        self.priors_mu = np.nan
        self.b_sigma = np.nan
        self.kesi = np.nan
        self.nu = np.nan

    def standardize_feature_vectors(self):
        self.feat = (self.feat - np.nanmean(self.feat, axis=0)) / np.nanstd(self.feat, axis=0)

    def normalize_feature_vectors(self):
        self.feat = (self.feat - np.nanmin(self.feat, axis=0)) / (np.nanmax(self.feat, axis=0) -
                                                                  np.nanmin(self.feat, axis=0))

    def calc_gibbs_energy(self, labels, beta):
        """Calculates the Gibbs energy for each element using the granular coefficient(s) beta.

        Args:
            labels (:obj:`np.ndarray`): the list of labels assigned to each element
            beta (:obj:'float' or `list` of float): if  len(beta) == 1, use isotropic Potts model or 1D scenario, else,
            use anisotropic Potts model.

        Returns:
            :obj:`np.ndarray` : Gibbs energy at every element for each label.
        """
        # ************************************************************************************************
        # 1D
        if self.phyDim == 1:
            # tile
            lt = np.tile(labels, (self.n_labels, 1)).T

            ge = np.arange(self.n_labels)  # elements x labels
            ge = np.tile(ge, (len(labels), 1)).astype(float)

            # first row
            top = np.expand_dims(np.not_equal(np.arange(self.n_labels), lt[1, :]) * beta, axis=0)
            # mid
            mid = (np.not_equal(ge[1:-1, :], lt[:-2, :]).astype(float) + np.not_equal(ge[1:-1, :], lt[2:, :]).astype(
                float)) * beta
            # last row
            bot = np.expand_dims(np.not_equal(np.arange(self.n_labels), lt[-2, :]) * beta, axis=0)
            # put back together and return gibbs energy
            return np.concatenate((top, mid, bot))

        # ************************************************************************************************
        # 2D
        elif self.phyDim == 2:

            # reshape the labels to 2D for "stencil-application"
            label_image = labels.reshape(self.phys_shp[0], self.phys_shp[1])

            # prepare gibbs energy array (filled with zeros)
            ref_matrix = np.empty((self.phys_shp[0] + 2, self.phys_shp[1] + 2,))
            ref_matrix[:] = np.nan
            ref_matrix[1:self.phys_shp[0] + 1, 1:self.phys_shp[1] + 1] = label_image
            ref_matrix_deck = np.tile(ref_matrix, (self.n_labels, 1, 1)).astype(float)

            # create comparison array containing the different labels
            comp_deck = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            for i in range(self.n_labels):
                comp_deck[i, :, :] = i

            # calculate left neighbor energy
            left = ref_matrix_deck[:, 1:self.phys_shp[0] + 1, 0:self.phys_shp[1]]
            diff = comp_deck - left
            temp_1 = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            temp_1[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate right neighbor energy (isotropic)
            right = ref_matrix_deck[:, 1:self.phys_shp[0] + 1, 2:self.phys_shp[1] + 2]
            diff = comp_deck - right
            temp_2 = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            temp_2[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate top neighbor energy (isotropic)
            top = ref_matrix_deck[:, 0:self.phys_shp[0], 1:self.phys_shp[1] + 1]
            diff = comp_deck - top
            temp_3 = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            temp_3[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate bottom neighbor energy (isotropic)
            bottom = ref_matrix_deck[:, 2:self.phys_shp[0] + 2, 1:self.phys_shp[1] + 1]
            diff = comp_deck - bottom
            temp_4 = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            temp_4[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate upper-left neighbor energy (isotropic)
            upper_left = ref_matrix_deck[:, 0:self.phys_shp[0], 0:self.phys_shp[1]]
            diff = comp_deck - upper_left
            temp_5 = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            temp_5[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate upper-right neighbor energy (isotropic)
            upper_right = ref_matrix_deck[:, 0:self.phys_shp[0], 2:self.phys_shp[1] + 2]
            diff = comp_deck - upper_right
            temp_6 = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            temp_6[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate lower-left neighbor energy (isotropic)
            lower_left = ref_matrix_deck[:, 2:self.phys_shp[0] + 2, 0:self.phys_shp[1]]
            diff = comp_deck - lower_left
            temp_7 = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            temp_7[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate lower-right neighbor energy (isotropic)
            lower_right = ref_matrix_deck[:, 2:self.phys_shp[0] + 2, 2:self.phys_shp[1] + 2]
            diff = comp_deck - lower_right
            temp_8 = np.tile(np.zeros_like(label_image), (self.n_labels, 1, 1)).astype(float)
            temp_8[(diff != 0) & (~np.isnan(diff))] = 1

            # multiply beta
            if self.beta_dim == 1:
                ge = (temp_1 + temp_2 + temp_3 + temp_4 + temp_5 + temp_6 + temp_7 + temp_8) * beta
            elif self.beta_dim == 4:
                # #  3  1  2
                # #   \ | /
                # #   --+-- 0
                # #   / | \

                ge = (temp_1 + temp_2)*beta[0] + \
                     (temp_3 + temp_4)*beta[1] + \
                     (temp_5 + temp_8)*beta[3] + \
                     (temp_6 + temp_7)*beta[2]
            else:
                raise Exception("Other beta configurations are not supported")

        elif self.phyDim == 3:
            # TODO: [3D] implementation of gibbs energy
            raise Exception("3D not yet implemented.")

        else:
            raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")

        # reshape and transpose gibbs energy, return
        ge_list = np.array([ge[l, :, :].ravel() for l in range(self.n_labels)]).T
        return ge_list

    def calc_like_energy(self, mu, cov):
        """Calculates the energy likelihood for a given mean array and covariance matrix for the entire domain.

        Args:
            mu (:obj:`np.ndarray`):
            cov (:obj:`np.ndarray`):

        Returns:
            :obj:`np.ndarray` : Energy likelihood for each label at each element.
        """
        le = np.zeros((self.num_pixels, self.n_labels))

        # uses flattened features array
        for l in range(self.n_labels):
            le[:, l] = np.einsum("...i,ji,...j",
                                 0.5 * np.array([self.feat - mu[l, :]]),
                                 np.linalg.inv(cov[l, :, :]),
                                 np.array([self.feat - mu[l, :]])) + 0.5 * np.log(
                                 np.linalg.det(cov[l, :, :]))

        return le

    def propose_beta(self, beta_prev, beta_jump_length):
        """Proposes a perturbed beta based on a jump length hyperparameter.

        Args:
            beta_prev:
            beta_jump_length:

        Returns:

        """
        # create proposal covariance depending on physical dimensionality
        # Possible beta_dim values are in [1, 4, 13]

        sigma_prop = np.eye(self.beta_dim) * beta_jump_length
        return multivariate_normal(mean=beta_prev, cov=sigma_prop).rvs()

    def propose_mu(self, mu_prev, mu_jump_length):
        """Proposes a perturbed mu matrix using a jump length hyperparameter.

        Args:
            mu_prev (:obj:`np.ndarray`): Previous mean array for all labels and features
            mu_jump_length (float or int): Hyperparameter specifying the jump length for the new proposal mean array.

        Returns:
            :obj:`np.ndarray`: The newly proposed mean array.

        """
        # prepare matrix
        mu_prop = np.ones((self.n_labels, self.n_feat))
        # loop over labels
        for l in range(self.n_labels):
            mu_prop[l, :] = multivariate_normal(mean=mu_prev[l, :], cov=np.eye(self.n_feat) * mu_jump_length).rvs()
        return mu_prop

    def log_prior_density_mu(self, mu, label):
        """Calculates the summed log prior density for a given mean and labels array."""
        with np.errstate(divide='ignore'):
            return np.sum(np.log(self.priors_mu[label].pdf(mu)))

    def log_prior_density_beta(self, beta):
        """Calculates the log prior density for a given beta array."""
        return np.log(self.prior_beta.pdf(beta))

    def log_prior_density_cov(self, cov, label):
        """Calculates the summed log prior density for the given covariance matrix and labels array."""
        lam = np.sqrt(np.diag(cov[label, :, :]))
        r = np.diag(1. / lam) @ cov[label, :, :] @ np.diag(1. / lam)
        logp_r = -0.5 * (self.nu + self.n_feat + 1) * np.log(np.linalg.det(r)) - self.nu / 2. * np.sum(
            np.log(np.diag(np.linalg.inv(r))))
        logp_lam = np.sum(np.log(multivariate_normal(mean=self.b_sigma[label, :],
                                                     cov=self.kesi[label, :]).pdf(np.log(lam.T))))
        return logp_r + logp_lam

    def calc_sum_log_mixture_density(self, comp_coef, mu, cov):
        """Calculate sum of log mixture density with each observation at every element.

        Args:
            comp_coef (:obj:`np.ndarray`): Component coefficient for each element (row) and label (column).
            mu (:obj:`np.ndarray`): Mean value array for all labels and features.
            cov (:obj:`np.ndarray`): Covariance matrix.

        Returns:
            float: Summed log mixture density.

        """
        lmd = np.zeros((self.phys_shp.prod(), self.n_labels))

        for l in range(self.n_labels):
            draw = multivariate_normal(mean=mu[l, :], cov=cov[l, :, :]).pdf(self.feat)
            multi = comp_coef[:, l] * np.array([draw])
            lmd[:, l] = multi
        lmd = np.sum(lmd, axis=1)
        with np.errstate(divide='ignore'):
            lmd = np.log(lmd)

        return np.sum(lmd)

    def gibbs_sample(self, t, beta_jump_length, mu_jump_length, cov_volume_jump_length, theta_jump_length, verbose,
                     fix_beta):
        """Takes care of the Gibbs sampling. This is the main function of the algorithm.

        Args:
            t: Hyperparameter
            beta_jump_length: Hyperparameter
            mu_jump_length: Hyperparameter
            cov_volume_jump_length: Hyperparameter
            theta_jump_length: Hyperparameter
            verbose (bool or :obj:`str`): Toggles verbosity.
            fix_beta (bool): Fixed beta to the inital value if True, else adaptive.

        Returns:
            The function updates directly on the object variables and appends new draws of labels and
            parameters to their respective storages.
        """
        # TODO: [GENERAL] In-depth description of the gibbs sampling function

        # ************************************************
        # CALCULATE TOTAL ENERGY
        # 1 - calculate energy likelihood for each element and label
        like_energy = self.calc_like_energy(self.mus[-1], self.covs[-1])
        if verbose == "energy":
            print("likelihood energy:", like_energy)
        # 2 - calculate gibbs/mrf energy
        gibbs_energy = self.calc_gibbs_energy(self.labels[-1], self.betas[-1])
        if verbose == "energy":
            print("gibbs energy:", gibbs_energy)
        # 3 - calculate total energy
        # total energy matrix 2d: n_elements by n_labels
        total_energy = like_energy + gibbs_energy
        if verbose == "energy":
            print("total_energy:", total_energy)
        # ************************************************************************************************
        # CALCULATE PROBABILITY OF LABELS
        labels_prob = _calc_labels_prob(total_energy, t)
        if verbose == "energy":
            print("Labels probability:", labels_prob)

        # append total energy matrix
        self.storage_te.append(total_energy)

        # make copy of previous labels
        new_labels = copy(self.labels[-1])

        for i, color_f in enumerate(self.colors):
            new_labels[color_f] = draw_labels_vect(labels_prob[color_f])
            # now recalculate gibbs energy and other energies from the mixture of old and new labels
            if i < 3:
                gibbs_energy = self.calc_gibbs_energy(new_labels, self.betas[-1])
                total_energy = like_energy + gibbs_energy  # + self_energy
                labels_prob = _calc_labels_prob(total_energy, t)

        # append labels generated from the current iteration
        self.labels.append(new_labels)

        # ************************************************************************************************
        # calculate energy for component coefficient
        energy_for_comp_coef = gibbs_energy

        # ************************************************************************************************
        # CALCULATE COMPONENT COEFFICIENT
        comp_coef = _calc_labels_prob(energy_for_comp_coef, t)

        # ************************************************************************************************
        # PROPOSAL STEP
        # make proposals for beta, mu and cov
        # beta depends on physical dimensions, for 1d its size 1
        beta_prop = self.propose_beta(self.betas[-1], beta_jump_length)
        # print("beta prop:", beta_prop)
        mu_prop = self.propose_mu(self.mus[-1], mu_jump_length)
        # print("mu prop:", mu_prop)
        cov_prop = _propose_cov(self.covs[-1], self.n_feat, self.n_labels, cov_volume_jump_length, theta_jump_length)
        # print("cov_prop:", cov_prop)

        # ************************************************************************************************
        # Compare mu, cov and beta proposals with previous, then decide which to keep for next iteration

        # prepare next ones
        mu_next = copy(self.mus[-1])
        cov_next = copy(self.covs[-1])
        beta_next = copy(self.betas[-1])

        # ************************************************************************************************
        # UPDATE MU
        for l in range(self.n_labels):
            # log-prob prior density for mu
            mu_temp = copy(mu_next)
            mu_temp[l, :] = mu_prop[l, :]

            lp_mu_prev = self.log_prior_density_mu(mu_next, l)
            lp_mu_prop = self.log_prior_density_mu(mu_temp, l)

            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_next)
            # calculate log mixture density for proposed mu and cov
            lmd_prop = self.calc_sum_log_mixture_density(comp_coef, mu_temp, cov_next)

            # combine
            log_target_prev = lmd_prev + lp_mu_prev
            log_target_prop = lmd_prop + lp_mu_prop

            mu_eval = evaluate(log_target_prop, log_target_prev)
            if mu_eval[0]:
                mu_next[l, :] = mu_prop[l, :]
            else:
                pass
            self.mu_acc_ratio = np.append(self.mu_acc_ratio, mu_eval[1])

        # append mu
        self.mus.append(mu_next)

        # ************************************************************************************************
        # UPDATE COVARIANCE
        for l in range(self.n_labels):
            cov_temp = copy(cov_next)
            cov_temp[l, :, :] = cov_prop[l, :, :]

            # log-prob prior density for covariance
            lp_cov_prev = self.log_prior_density_cov(cov_next, l)
            # print("lp_cov_prev:", lp_cov_prev)
            lp_cov_prop = self.log_prior_density_cov(cov_temp, l)
            # print("lp_cov_prop:", lp_cov_prop)

            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_next)
            # print("lmd_prev:", lmd_prev)
            # calculate log mixture density for proposed mu and cov
            lmd_prop = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_temp)
            # print("lmd_prop:", lmd_prop)

            # combine
            log_target_prev = lmd_prev + lp_cov_prev
            log_target_prop = lmd_prop + lp_cov_prop

            cov_eval = evaluate(log_target_prop, log_target_prev)
            if cov_eval[0]:
                cov_next[l, :] = cov_prop[l, :]
            else:
                pass
            self.cov_acc_ratio = np.append(self.cov_acc_ratio, cov_eval[1])

        # append cov
        self.covs.append(cov_next)

        if not fix_beta:
            # ************************************************************************************************
            # UPDATE BETA
            lp_beta_prev = self.log_prior_density_beta(self.betas[-1])
            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, self.mus[-1], self.covs[-1])

            if self.beta_dim == 1:
                # calculate gibbs energy with new labels and proposed beta
                energy_for_comp_coef_prop = self.calc_gibbs_energy(self.labels[-1], beta_prop)
                comp_coef_prop = _calc_labels_prob(energy_for_comp_coef_prop, t)

                lmd_prop = self.calc_sum_log_mixture_density(comp_coef_prop, self.mus[-1], self.covs[-1])
                lp_beta_prop = self.log_prior_density_beta(beta_prop)

                log_target_prev = lmd_prev + lp_beta_prev
                # print("log_target_prev:", log_target_prev)
                log_target_prop = lmd_prop + lp_beta_prop
                # print("log_target_prop:", log_target_prop)

                beta_eval = evaluate(log_target_prop, log_target_prev)
                if beta_eval[0]:
                    beta_next = beta_prop
                else:
                    pass

                self.beta_acc_ratio = np.append(self.beta_acc_ratio, beta_eval[1])  # store
                self.betas.append(beta_next)

            else:
                for i_beta in range(self.beta_dim):
                    beta_temp = copy(beta_next)
                    beta_temp[i_beta] = beta_prop[i_beta]
                    # calculate gibbs energy with new labels and proposed beta
                    energy_for_comp_coef_prop = self.calc_gibbs_energy(self.labels[-1], beta_temp)
                    comp_coef_prop = _calc_labels_prob(energy_for_comp_coef_prop, t)
                    lmd_prop = self.calc_sum_log_mixture_density(comp_coef_prop, self.mus[-1], self.covs[-1])
                    lp_beta_prop = self.log_prior_density_beta(beta_temp)
                    log_target_prev = lmd_prev + lp_beta_prev
                    log_target_prop = lmd_prop + lp_beta_prop

                    beta_eval = evaluate(log_target_prop, log_target_prev)
                    if beta_eval[0]:
                        beta_next[i_beta] = beta_prop[i_beta]
                    else:
                        pass

                    self.beta_acc_ratio = np.append(self.beta_acc_ratio, beta_eval[1])  # store

                self.betas.append(beta_next)

        else:
            self.betas.append(beta_next)

            # ************************************************************************************************

    def fit(self, n, n_labels, beta_init=1, beta_jump_length=0.1, mu_jump_length=0.0005, cov_volume_jump_length=0.00005,
            theta_jump_length=0.0005, t=1., tol=5e-5, reg_covar=1e-3, max_iter=1000, n_init=100,
            verbose=False, fix_beta=False):
        """Fit the segmentation parameters to the given data.

        Args:
            n (int): Number of iterations.
            n_labels (int): Number of labels representing the number of clusters to be segmented.
            beta_init (float): Initial penalty value for Gibbs energy calculation.
            beta_jump_length (float): Hyperparameter specifying the beta proposal jump length.
            mu_jump_length (float): Hyperparameter for the mean proposal jump length.
            cov_volume_jump_length (float):
            theta_jump_length (float):
            t (float):
            tol (float): tolerance of difference at converge
            reg_covar (float): regularization value of covariance matrix
            max_iter (int): max number of iteration
            n_init (int): number of initial trials
            verbose (bool or :obj:`str`):
            fix_beta (bool):

        """
        # ************************************************************************************************
        # INIT GAUSSIAN MIXTURE MODEL
        # store n_labels
        print('Fitting the initial Gaussian mixture model...')
        self.n_labels = n_labels
        self.gmm = mixture.GaussianMixture(n_components=n_labels,
                                           covariance_type='full',
                                           tol=tol,
                                           reg_covar=reg_covar,
                                           max_iter=max_iter,
                                           n_init=n_init)
        self.gmm.fit(self.feat)

        # do initial prediction based on fit and observations, store as first entry in labels
        # ************************************************************************************************
        # INIT LABELS, MU and COV based on GMM
        self.labels = [self.gmm.predict(self.feat)]  # return the MAP estimate
        # INIT MU (mean from initial GMM)
        self.mus = [self.gmm.means_]
        # INIT COV (covariances from initial GMM)
        self.covs = [self.gmm.covariances_]

        # ************************************************************************************************
        # Initialize PRIOR distributions for beta, mu and covariance
        # BETA
        if self.phyDim == 1:
            self.beta_dim = 1
            self.betas = [beta_init]
            self.prior_beta = norm(beta_init, np.eye(1) * 100)
        elif self.phyDim == 2:
            if len(np.shape([beta_init])) == 1:
                # using isotropic Potts model
                self.beta_dim = 1
                self.betas = [beta_init]
                self.prior_beta = norm(beta_init, np.eye(1) * 100)
            else:
                # using anisotropic Potts model
                self.beta_dim = 4
                self.betas = [beta_init]
                self.prior_beta = multivariate_normal(beta_init, np.eye(self.beta_dim) * 100)
        elif self.phyDim == 3:
            raise Exception("3D not yet supported.")
        else:
            raise Exception("higher dimensional physical space (more than 3-D) not yet supported.")

        # MU
        # generate distribution means for each label
        prior_mu_means = [self.mus[0][label] for label in range(self.n_labels)]
        # generate distribution covariances for each label
        prior_mu_stds = [np.eye(self.n_feat) * 100 for label in range(self.n_labels)]
        # use the above to generate multivariate normal distributions for each label
        self.priors_mu = [multivariate_normal(prior_mu_means[label], prior_mu_stds[label]) for label in
                          range(self.n_labels)]

        # COV
        # generate b_sigma
        self.b_sigma = np.zeros((self.n_labels, self.n_feat))
        for l in range(self.n_labels):
            self.b_sigma[l, :] = np.log(np.sqrt(np.diag(self.gmm.covariances_[l, :, :])))
        # generate kesi
        self.kesi = np.ones((self.n_labels, self.n_feat)) * 100
        # generate nu
        self.nu = self.n_feat + 1

        print('Fitting is done!')
        # ************************************************************************************************

        for g in tqdm.trange(n):
            self.gibbs_sample(t, beta_jump_length, mu_jump_length, cov_volume_jump_length, theta_jump_length,
                              verbose, fix_beta)

    def get_estimator(self, start_iter):
        est = estimator(self.mus, self.covs, self.betas, start_iter)
        self.mu_est = est[0]
        self.mu_std = est[1]
        self.cov_est = est[2]
        self.cov_std = est[3]
        self.beta_est = est[4]
        self.beta_std = est[5]

    def get_label_prob(self, start_iter):
        self.label_prob = np.full((self.n_labels, self.num_pixels), np.nan)
        label_bin = np.array(self.labels)[start_iter:, :]
        for i in range(self.n_labels):
            count_i = np.sum(label_bin == i, axis=0)
            self.label_prob[i, :] = count_i / label_bin.shape[0]

    def get_map(self):
        # calculate MAP of labels. NOTE: get_label_prob must be executed first
        self.label_map_est = np.argmax(self.label_prob, axis=0)

    def get_ie(self):
        # calculate information entropy. NOTE: get_label_prob must be executed first
        temp = np.copy(self.label_prob)
        temp[np.where(temp == 0)] = 1
        self.info_entr = np.sum(-temp * np.log(temp), axis=0)


def pseudocolor(physic_shape, stencil=None):
    """Graph coloring based on the physical dimensions for independent labels draw.

    Args:
        physic_shape (:obj:`tuple` of int): physical shape of the data structure.
        stencil: the type of neighborhood

    Returns:
        1-DIMENSIONAL:
        return  color: graph color vector for parallel Gibbs sampler
        2-DIMENSIONAL:
        return  color: graph color vector for parallel Gibbs sampler
        return  colored image
    """

    dim = len(physic_shape)
    # ************************************************************************************************
    # 1-DIMENSIONAL
    if dim == 1:
        i_w = np.arange(0, physic_shape[0], step=2)
        i_b = np.arange(1, physic_shape[0], step=2)

        return np.array([i_w, i_b])

    # ************************************************************************************************
    # 2-DIMENSIONAL
    elif dim == 2:
        if stencil is None or stencil == "8p":
            # use 8 stamp as default, resulting in 4 colors
            num_of_colors = 4
            # color image
            colored_image = np.tile(np.kron([[0, 1], [2, 3]] * int(math.ceil(physic_shape[0] / 2)), np.ones((1, 1))),
                                    math.ceil(physic_shape[1] / 2))[0:physic_shape[0], 0:physic_shape[1]]
            colored_flat = colored_image.reshape(physic_shape[0] * physic_shape[1])

            # initialize storage array
            ci = []
            for c in range(num_of_colors):
                x = np.where(colored_flat == c)[0]
                ci.append(x)
            return np.array(ci)

        else:
            raise Exception(" In 2D space the stamp parameter needs to be either None (defaults to 8p)")

    # ************************************************************************************************
    # 3-DIMENSIONAL
    elif dim == 3:
        raise Exception("3D space not yet supported.")
        # TODO: 3d graph coloring

    else:
        raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")


def _calc_labels_prob(te, t):
    """"Calculate labels probability for array of total energies (te) and totally arbitrary skalar value t."""
    return (np.exp(-te / t).T / np.sum(np.exp(-te / t), axis=1)).T


def draw_labels_vect(labels_prob):
    """Vectorized draw of the label for each elements respective labels probability.

    Args:
        labels_prob (:obj:`np.ndarray`): (n_elements x n_labels) ndarray containing the element-specific labels
            probabilities for each element.

    Returns:
        :obj:`np.array` : Flat array containing the newly drawn labels for each element.

    """

    # cumsum labels probabilities for each element
    p = np.cumsum(labels_prob, axis=1)
    p = np.concatenate((np.zeros((p.shape[0], 1)), p), axis=1)

    # draw a random number between 0 and 1 for each element
    r = np.array([np.random.rand(p.shape[0])]).T
    # print(r)

    # compare and count to get label
    temp = np.sum(np.greater_equal((r @ np.ones((1, p.shape[1])) - p), 0), axis=1) - 1

    return temp


def _propose_cov(cov_prev, n_feat, n_labels, cov_jump_length, theta_jump_length):
    """Proposes a perturbed n-dimensional covariance matrix based on an existing one and a covariance jump length and
    theta jump length parameter.

    Args:
        cov_prev (:obj:`np.ndarray`): Covariance matrix.
        n_feat (int): Number of features.
        n_labels (int): Number of labels.
        cov_jump_length (float): Hyperparameter
        theta_jump_length (float): Hyperparameter

    Returns:
        :obj:`np.ndarray` : Perturbed covariance matrix.

    """
    # do svd on the previous covariance matrix
    comb = list(combinations(range(n_feat), 2))
    n_comb = len(comb)
    theta_jump = multivariate_normal(mean=[0 for i in range(n_comb)], cov=np.ones(n_comb) * theta_jump_length).rvs()

    if n_comb == 1:  # turn it into a list if there is only one combination (^= 2 features)
        theta_jump = [theta_jump]

    cov_prop = np.zeros_like(cov_prev)
    # print("cov_prev:", cov_prev)

    # loop over all labels (=layers of the covariance matrix)
    for l in range(n_labels):
        v_l, d_l, v_l_t = np.linalg.svd(cov_prev[l, :, :])

        # generate d jump
        log_d_jump = multivariate_normal(mean=[0 for i in range(n_feat)], cov=np.eye(n_feat) * cov_jump_length).rvs()
        # sum towards d proposal
        d_prop = np.diag(np.exp(np.log(d_l) + log_d_jump))
        # now tackle generating v jump
        a = np.eye(n_feat)
        for val in range(n_comb):
            rotation_matrix = _cov_proposal_rotation_matrix(v_l[:, comb[val][0]], v_l[:, comb[val][1]], theta_jump[val])
            a = rotation_matrix @ a

        v_prop = a @ v_l
        cov_prop[l, :, :] = v_prop @ d_prop @ v_prop.T
    return cov_prop


def _cov_proposal_rotation_matrix(x, y, theta):
    """Creates the rotation matrix needed for the covariance matrix proposal step.

    Args:
        x (:obj:`np.array`): First base vector.
        y (:obj:`np.array`): Second base vector.
        theta (float): Rotation angle.

    Returns:
        :obj:`np.ndarray` : Rotation matrix for covariance proposal step.

    """
    x = np.array([x]).T
    y = np.array([y]).T

    uu = x / np.linalg.norm(x)
    vv = y - uu.T @ y * uu
    vv = vv / np.linalg.norm(vv)
    # what is happening
    rotation_matrix = np.eye(len(x)) - uu @ uu.T - vv @ vv.T + np.hstack((uu, vv)) @ np.array(
        [[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.hstack((uu, vv)).T
    return rotation_matrix


def evaluate(log_target_prop, log_target_prev):

    ratio = np.exp(np.longfloat(log_target_prop - log_target_prev))
    ratio = min(ratio, 1)

    if (ratio == 1) or (np.random.uniform() < ratio):
        return True, ratio  # if accepted

    else:
        return False, ratio  # if rejected


def estimator(mus, covs, betas, start_iter):
    mus = np.array(mus)
    covs = np.array(covs)
    betas = np.array(betas)
    mu_est = np.mean(mus[start_iter:, :], axis=0)
    mu_std = np.std(mus[start_iter:, :], axis=0)
    cov_est = np.mean(covs[start_iter:, :], axis=0)
    cov_std = np.std(covs[start_iter:, :], axis=0)
    if len(betas.shape) == 1:
        beta_est = np.mean(betas[start_iter:])
        beta_std = np.std(betas[start_iter:])
    else:
        beta_est = np.mean(betas[start_iter:, :], axis=0)
        beta_std = np.std(betas[start_iter:, :], axis=0)
    return mu_est, mu_std, cov_est, cov_std, beta_est, beta_std
