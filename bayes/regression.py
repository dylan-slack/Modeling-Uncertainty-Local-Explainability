"""Bayesian regression.

A class the implements the Bayesian Regression.
"""
import operator as op
from functools import reduce
import copy
import collections

import numpy as np
from scipy.stats import invgamma 
from scipy.stats import multivariate_normal

class BayesianLinearRegression:
    def __init__(self, percent=95, l2=True, prior=None): 
        if prior is not None:
            raise NameError("Currently only support uninformative prior, set to None plz.")
    
        self.percent = percent
        self.l2 = l2

    def fit(self, xtrain, ytrain, sample_weight, compute_creds=True):
        """
        Fit the bayesian linear regression.
        
        Arguments:
            xtrain: the training data
            ytrain: the training labels
            sample_weight: the weights for fitting the regression
        """

        # store weights
        weights = sample_weight

        # add intercept
        xtrain = np.concatenate((np.ones(xtrain.shape[0])[:,None], xtrain), axis=1)
        diag_pi_z = np.zeros((len(weights), len(weights)))
        np.fill_diagonal(diag_pi_z, weights)

        if self.l2:
            V_Phi = np.linalg.inv(xtrain.transpose().dot(diag_pi_z).dot(xtrain) \
                            + np.eye(xtrain.shape[1]))
        else:
            V_Phi = np.linalg.inv(xtrain.transpose().dot(diag_pi_z).dot(xtrain))

        Phi_hat = V_Phi.dot(xtrain.transpose()).dot(diag_pi_z).dot(ytrain)

        N = xtrain.shape[0]
        Y_m_Phi_hat = ytrain - xtrain.dot(Phi_hat)

        s_2 = (1.0 / N) * (Y_m_Phi_hat.dot(diag_pi_z).dot(Y_m_Phi_hat) \
                     + Phi_hat.transpose().dot(Phi_hat))

        self.score = s_2

        self.s_2 = s_2
        self.N = N
        self.V_Phi = V_Phi
        self.Phi_hat = Phi_hat
        self.coef_ = Phi_hat[1:]
        self.intercept_ = Phi_hat[0]
        self.weights = weights

        if compute_creds:
            self.creds = self.get_creds(percent=self.percent)
        else:
            self.creds = "NA"

        self.crit_params = {
            "s_2": self.s_2,
            "N": self.N,
            "V_Phi": self.V_Phi,
            "Phi_hat": self.Phi_hat,
            "creds": self.creds
        }

        return self

    def predict(self, data):
        """
        The predictive distribution.

        Arguments:
            data: The data to predict
        """
        q_1 = np.eye(data.shape[0])
        data_ones = np.concatenate((np.ones(data.shape[0])[:,None], data), axis=1)

        # Get response
        response = np.matmul(data, self.coef_)
        response += self.intercept_

        # Compute var
        temp = np.matmul(data_ones, self.V_Phi)
        mat = np.matmul(temp, data_ones.transpose())
        var = self.s_2 * (q_1 + mat)
        diag = np.diagonal(var)
        
        return response, np.sqrt(diag)

    def get_ptg(self, desired_width):
        """
        Compute the ptg perturbations.
        """
        cert = (desired_width / 1.96) ** 2
        S = self.coef_.shape[0] * self.s_2
        T = np.mean(self.weights)
        return 4 * S / (self.coef_.shape[0] * T * cert)

    def get_creds(self, percent=95, n_samples=10_000, get_intercept=False):
        """
        Get the credible intervals.

        Arguments:
            percent: the percent cutoff for the credible interval, i.e., 95 is 95% credible interval
            n_samples: the number of samples to compute the credible interval
            get_intercept: whether to include the intercept in the credible interval
        """
        samples = self.draw_posterior_samples(n_samples, get_intercept=get_intercept)
        creds = np.percentile(np.abs(samples - (self.Phi_hat if get_intercept else self.coef_)),
                              percent,
                              axis=0)
        return creds

    def draw_posterior_samples(self, num_samples, get_intercept=False):
        """
        Sample from the posterior.
        
        Arguments:
            num_samples: number of samples to draw from the posterior
            get_intercept: whether to include the intercept
        """

        sigma_2 = invgamma.rvs(self.N / 2, scale=(self.N * self.s_2) / 2, size=num_samples)

        phi_samples = []
        for sig in sigma_2:
            sample = multivariate_normal.rvs(mean=self.Phi_hat, 
                                             cov=self.V_Phi * sig, 
                                             size=1)
            phi_samples.append(sample)

        phi_samples = np.vstack(phi_samples)

        if get_intercept:
            return phi_samples
        else:
            return phi_samples[:, 1:]