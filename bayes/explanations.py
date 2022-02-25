"""Bayesian Local Explanations.

This code implements bayesian local explanations. The code supports the LIME & SHAP 
kernels. Along with the LIME & SHAP feature importances, bayesian local explanations
also support uncertainty expression over the feature importances.
"""
import logging

from copy import deepcopy
from functools import reduce
from multiprocessing import Pool
import numpy as np
import operator as op
from tqdm import tqdm

import sklearn
import sklearn.preprocessing
from sklearn.linear_model import Ridge, Lasso
from lime import lime_image, lime_tabular

from bayes.regression import BayesianLinearRegression

LDATA, LINVERSE, LSCALED, LDISTANCES, LY = list(range(5))
SDATA, SINVERSE, SY = list(range(3))

class BayesLocalExplanations:
    """Bayesian Local Explanations.

    This class implements the bayesian local explanations.
    """
    def __init__(self,
                 training_data,
                 data="image",
                 kernel="lime",
                 credible_interval=95,
                 mode="classification",
                 categorical_features=[],
                 discretize_continuous=True,
                 save_logs=False,
                 log_file_name="bayes.log",
                 width=0.75,
                 verbose=False):
        """Initialize the local explanations.

        Arguments:
            training_data: The 
            data: The type of data, either "image" or "tabular"
            kernel: The kernel to use, either "lime" or "shap"
            credible_interval: The % credible interval to use for the feature importance
                               uncertainty.
            mode: Whether to run with classification or regression.
            categorical_features: The indices of the categorical features, if in regression mode.
            save_logs: Whether to save logs from the run.
            log_file_name: The name of log file.
        """

        assert kernel in ["lime", "shap"], f"Kernel must be one of lime or shap, not {kernel}"
        assert data in ["image", "tabular"], f"Data must be one of image or tabular, not {data}"
        assert mode in ["classification"], "Others modes like regression are not implemented"

        if save_logs:
            logging.basicConfig(filename=log_file_name,
                                filemode='a',
                                level=logging.INFO)

        logging.info("==============================================")
        logging.info("Initializing Bayes%s %s explanations", kernel, data)
        logging.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.cred_int = credible_interval
        self.data = data
        self.kernel = kernel
        self.mode = mode
        self.categorical_features = categorical_features
        self.discretize_continuous = discretize_continuous
        self.verbose = verbose
        self.width = width * np.sqrt(training_data.shape[1])

        logging.info("Setting mode to %s", mode)
        logging.info("Credible interval set to %s", self.cred_int)

        if kernel == "shap" and data == "tabular":
            logging.info("Setting discretize_continuous to True, due to shapley sampling")
            discretize_continuous = True

        self.training_data = training_data
        self._run_init(training_data)

    def _run_init(self, training_data):
        if self.kernel == "lime":
            lime_tab_exp = lime_tabular.LimeTabularExplainer(training_data, 
                                                             mode=self.mode,
                                                             categorical_features=self.categorical_features,
                                                             discretize_continuous=self.discretize_continuous)
            self.lime_info = lime_tab_exp
        elif self.kernel == "shap":
            # Discretization forcibly set to true for shap sampling on initialization
            shap_tab_exp = lime_tabular.LimeTabularExplainer(training_data, 
                                                             mode=self.mode,
                                                             categorical_features=self.categorical_features,
                                                             discretize_continuous=self.discretize_continuous)
            self.shap_info = shap_tab_exp
        else:
            raise NotImplementedError

    def _log_args(self, args):
        """Logs arguments to function."""
        logging.info(args)

    def _shap_tabular_perturb_n_samples(self,
                                        data,
                                        n_samples,
                                        max_coefs=None):
        """Generates n shap perturbations"""
        if max_coefs is None:
            max_coefs = np.arange(data.shape[0])
        pre_rdata, pre_inverse = self.shap_info._LimeTabularExplainer__data_inverse(data_row=data, 
                                                                                    num_samples=n_samples)
        rdata = pre_rdata[:, max_coefs]
        inverse = np.tile(data, (n_samples, 1))
        inverse[:, max_coefs] = pre_inverse[:, max_coefs]
        return rdata, inverse

    def _lime_tabular_perturb_n_samples(self,
                                        data,
                                        n_samples):
        """Generates n_perturbations for LIME."""
        rdata, inverse = self.lime_info._LimeTabularExplainer__data_inverse(data_row=data, 
                                                                            num_samples=n_samples)
        scaled_data = (rdata - self.lime_info.scaler.mean_) / self.lime_info.scaler.scale_
        distances = sklearn.metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric='euclidean'
        ).ravel()
        return rdata, inverse, scaled_data, distances

    def _stack_tabular_return(self, existing_return, perturb_return):
        """Stacks data from new tabular return to existing return."""
        if len(existing_return) == 0:
            return perturb_return
        new_return = []
        for i, item in enumerate(existing_return):
            new_return.append(np.concatenate((item, perturb_return[i]), axis=0))
        return new_return

    def _select_indices_from_data(self, perturb_return, indices, predictions):
        """Gets each element from the perturb return according to indices, then appends the predictions."""
        # Previoulsy had this set to range(4)
        temp = [perturb_return[i][indices] for i in range(len(perturb_return))]
        temp.append(predictions)
        return temp

    def shap_tabular_focus_sample(self,
                                  data, 
                                  classifier_f,
                                  label,
                                  n_samples,
                                  focus_sample_batch_size,
                                  focus_sample_initial_points,
                                  to_consider=10_000,
                                  tempurature=1e-2,
                                  enumerate_initial=True):
        """Focus sample n_samples perturbations for lime tabular."""
        assert focus_sample_initial_points > 0, "Initial focusing sample points cannot be <= 0"
        current_n_perturbations = 0

        # Get 1's coalitions, if requested
        if enumerate_initial:
            enumerate_init_p = self._enumerate_initial_shap(data)
            current_n_perturbations += enumerate_init_p[0].shape[0]
        else:
            enumerate_init_p = None

        if self.verbose:
            pbar = tqdm(total=n_samples)
            pbar.update(current_n_perturbations)

        # Get initial points
        if current_n_perturbations < focus_sample_initial_points:
            initial_perturbations = self._shap_tabular_perturb_n_samples(data, focus_sample_initial_points - current_n_perturbations)

            if enumerate_init_p is not None:
                current_perturbations = self._stack_tabular_return(enumerate_init_p, initial_perturbations)
            else:
                current_perturbations = initial_perturbations
                
            current_n_perturbations += initial_perturbations[0].shape[0]
        else:
            current_perturbations = enumerate_init_p
        
        current_perturbations = list(current_perturbations)
        
        # Store initial predictions
        current_perturbations.append(classifier_f(current_perturbations[SINVERSE])[:, label])
        if self.verbose:
            pbar.update(initial_perturbations[0].shape[0])
        
        while current_n_perturbations < n_samples:
            current_batch_size = min(focus_sample_batch_size, n_samples - current_n_perturbations)

            # Init current BLR
            blr = BayesianLinearRegression(percent=self.cred_int)
            weights = self._get_shap_weights(current_perturbations[SDATA], current_perturbations[SDATA].shape[1])
            blr.fit(current_perturbations[SDATA], current_perturbations[-1], weights, compute_creds=False)
            
            candidate_perturbations = self._shap_tabular_perturb_n_samples(data, to_consider)
            _, var = blr.predict(candidate_perturbations[SINVERSE])

            # Get sampling weighting
            var /= tempurature
            exp_var = np.exp(var)
            all_exp = np.sum(exp_var)
            tempurature_scaled_weights = exp_var / all_exp
            
            # Get sampled indices
            least_confident_sample = np.random.choice(len(var), size=current_batch_size, p=tempurature_scaled_weights, replace=True)

            # Get predictions
            cy = classifier_f(candidate_perturbations[SINVERSE][least_confident_sample])[:, label]

            new_perturbations = self._select_indices_from_data(candidate_perturbations, least_confident_sample, cy)
            current_perturbations = self._stack_tabular_return(current_perturbations, new_perturbations)
            current_n_perturbations += new_perturbations[0].shape[0]

            if self.verbose:
                pbar.update(new_perturbations[0].shape[0])

        return current_perturbations

    def lime_tabular_focus_sample(self,
                                  data, 
                                  classifier_f,
                                  label,
                                  n_samples,
                                  focus_sample_batch_size,
                                  focus_sample_initial_points,
                                  to_consider=10_000,
                                  tempurature=5e-4,
                                  existing_data=[]):
        """Focus sample n_samples perturbations for lime tabular."""
        current_n_perturbations = 0

        # Get initial focus sampling batch
        if len(existing_data) < focus_sample_initial_points:
            # If there's existing data, make sure we only sample up to existing_data points
            initial_perturbations = self._lime_tabular_perturb_n_samples(data, focus_sample_initial_points - len(existing_data))
            current_perturbations = self._stack_tabular_return(existing_data, initial_perturbations)
        else:
            current_perturbations = existing_data

        if self.verbose:
            pbar = tqdm(total=n_samples)
        
        current_perturbations = list(current_perturbations)
        current_n_perturbations += initial_perturbations[0].shape[0]
        
        # Store predictions on initial data
        current_perturbations.append(classifier_f(current_perturbations[LINVERSE])[:, label])
        if self.verbose:
            pbar.update(initial_perturbations[0].shape[0])
        
        # Sample up to n_samples 
        while current_n_perturbations < n_samples:

            # If batch size would exceed n_samples, only sample enough to reach n_samples
            current_batch_size = min(focus_sample_batch_size, n_samples - current_n_perturbations)

            # Init current BLR
            blr = BayesianLinearRegression(percent=self.cred_int)
            # Get weights on current distances
            weights = self._lime_kernel(current_perturbations[LDISTANCES], self.width)
            # Fit blr on current perturbations & data
            blr.fit(current_perturbations[LDATA], current_perturbations[LY], weights)
            
            # Get set of perturbations to consider labeling
            candidate_perturbations = self._lime_tabular_perturb_n_samples(data, to_consider)
            _, var = blr.predict(candidate_perturbations[LDATA])

            # Reweight
            var /= tempurature
            exp_var = np.exp(var)
            all_exp = np.sum(exp_var)
            tempurature_scaled_weights = exp_var / all_exp
            
            # Get sampled indices
            least_confident_sample = np.random.choice(len(var), size=current_batch_size, p=tempurature_scaled_weights, replace=False)

            # Get predictions
            cy = classifier_f(candidate_perturbations[LINVERSE][least_confident_sample])[:, label]

            new_perturbations = self._select_indices_from_data(candidate_perturbations, least_confident_sample, cy)
            current_perturbations = self._stack_tabular_return(current_perturbations, new_perturbations)
            current_n_perturbations += new_perturbations[0].shape[0]

            if self.verbose:
                pbar.update(new_perturbations[0].shape[0])

        return current_perturbations

    def _lime_kernel(self, d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    def _explain_bayes_lime(self,
                            data,
                            classifier_f,
                            label, 
                            focus_sample,
                            cred_width,
                            n_samples,
                            max_n_samples,
                            focus_sample_batch_size,
                            focus_sample_initial_points,
                            ptg_initial_points,
                            to_consider):
        """Computes the bayeslime tabular explanations."""

        # Case where only n_samples is specified and not focused sampling
        if n_samples is not None and not focus_sample:
            logging.info("Generating bayeslime explanation with %s samples", n_samples)

            # Generate perturbations
            rdata, inverse, scaled_data, distances = self._lime_tabular_perturb_n_samples(data, n_samples)
            weights = self._lime_kernel(distances, self.width)
            y = classifier_f(inverse)[:, label]
            blr = BayesianLinearRegression(percent=self.cred_int)
            blr.fit(rdata, y, weights)
        # Focus sampling
        elif focus_sample:
            logging.info("Starting focused sampling")
            if n_samples:
                logging.info("n_samples preset, running focused sampling up to %s samples", n_samples)
                logging.info("using batch size %s with %s initial points", focus_sample_batch_size, focus_sample_initial_points)
                focused_sampling_output = self.lime_tabular_focus_sample(data,
                                                                         classifier_f,
                                                                         label,
                                                                         n_samples,
                                                                         focus_sample_batch_size,
                                                                         focus_sample_initial_points,
                                                                         to_consider=to_consider,
                                                                         existing_data=[])
                rdata = focused_sampling_output[LDATA]
                distances = focused_sampling_output[LDISTANCES]
                y = focused_sampling_output[LY]

                blr = BayesianLinearRegression(percent=self.cred_int)
                weights = self._lime_kernel(distances, self.width)
                blr.fit(rdata, y, weights)
            else:
                # Use ptg to get the number of samples, then focus sample
                # Note, this isn't used in the paper, this case currently isn't implemented
                raise NotImplementedError

        else:
            # PTG Step 1, get initial
            rdata, inverse, scaled_data, distances = self._lime_tabular_perturb_n_samples(data, ptg_initial_points)
            weights = self._lime_kernel(distances, self.width)
            y = classifier_f(inverse)[:, label]
            blr = BayesianLinearRegression(percent=self.cred_int)
            blr.fit(rdata, y, weights)    

            # PTG Step 2, get additional points needed
            n_needed = int(np.ceil(blr.get_ptg(cred_width)))
            if self.verbose:
                tqdm.write(f"Additional Number of perturbations needed is {n_needed}")
            ptg_rdata, ptg_inverse, ptg_scaled_data, ptg_distances = self._lime_tabular_perturb_n_samples(data, n_needed - ptg_initial_points)
            ptg_weights = self._lime_kernel(ptg_distances, self.width)

            rdata = np.concatenate((rdata, ptg_rdata), axis=0)
            inverse = np.concatenate((inverse, ptg_inverse), axis=0)
            scaled_data = np.concatenate((scaled_data, ptg_scaled_data), axis=0)
            distances = np.concatenate((distances, ptg_distances), axis=0)

            # Run final model
            ptgy = classifier_f(ptg_inverse)[:, label]
            y = np.concatenate((y, ptgy), axis=0)
            blr = BayesianLinearRegression(percent=self.cred_int)
            blr.fit(rdata, y, self._lime_kernel(distances, self.width))    

        # Format output for returning
        output = {
            "data": rdata,
            "y": y,
            "distances": distances,
            "blr": blr,
            "coef": blr.coef_,
            "max_coefs": None # Included for consistency purposes w/ bayesshap
        }

        return output

    def _get_shap_weights(self, data, M):
        """Gets shap weights. This assumes data is binary."""
        nonzero = np.count_nonzero(data, axis=1)
        weights = []
        for nz in nonzero:
            denom = (nCk(M, nz) * nz * (M - nz))
            # Stabilize kernel
            if denom == 0:
                weight = 1.0
            else:
                weight = ((M - 1) / denom)
            weights.append(weight)
        return weights

    def _enumerate_initial_shap(self, data, max_coefs=None):
        """Enumerate 1's for stability."""
        if max_coefs is None:
            data = np.eye(data.shape[0])
            inverse = self.shap_info.discretizer.undiscretize(data)
            return data, inverse
        else:
            data = np.zeros((max_coefs.shape[0], data.shape[0]))
            for i in range(max_coefs.shape[0]):
                data[i, max_coefs[i]] = 1
            inverse = self.shap_info.discretizer.undiscretize(data)
            return data[:, max_coefs], inverse

    def _explain_bayes_shap(self,
                            data,
                            classifier_f,
                            label, 
                            focus_sample,
                            cred_width,
                            n_samples,
                            max_n_samples,
                            focus_sample_batch_size,
                            focus_sample_initial_points,
                            ptg_initial_points,
                            to_consider,
                            feature_select_num_points=1_000,
                            n_features=10,
                            l2=True,
                            enumerate_initial=True,
                            feature_selection=True,
                            max_coefs=None):
        """Computes the bayesshap tabular explanations."""
        if feature_selection and max_coefs is None:
            n_features = min(n_features, data.shape[0])
            _, feature_select_inverse = self._shap_tabular_perturb_n_samples(data, feature_select_num_points)
            lr = Ridge().fit(feature_select_inverse, classifier_f(feature_select_inverse)[:, label])
            max_coefs = np.argsort(np.abs(lr.coef_))[-1 * n_features:]
        elif feature_selection and max_coefs is not None:
            pass 
        else:
            max_coefs = None
             
        # Case without focused sampling
        if n_samples is not None and not focus_sample:
            logging.info("Generating bayesshap explanation with %s samples", n_samples)

            # Enumerate single coalitions, if requested
            if enumerate_initial:
                data_init, inverse_init = self._enumerate_initial_shap(data, max_coefs)
                n_more = n_samples - inverse_init.shape[0]
            else:
                n_more = n_samples

            rdata, inverse = self._shap_tabular_perturb_n_samples(data, n_more, max_coefs)

            if enumerate_initial:
                rdata = np.concatenate((data_init, rdata), axis=0)
                inverse = np.concatenate((inverse_init, inverse), axis=0)

            y = classifier_f(inverse)[:, label]
            weights = self._get_shap_weights(rdata, M=rdata.shape[1])

            blr = BayesianLinearRegression(percent=self.cred_int)
            blr.fit(rdata, y, weights)
        elif focus_sample:
            if feature_selection:
                raise NotImplementedError

            logging.info("Starting focused sampling")
            if n_samples:
                logging.info("n_samples preset, running focused sampling up to %s samples", n_samples)
                logging.info("using batch size %s with %s initial points", focus_sample_batch_size, focus_sample_initial_points)
                focused_sampling_output = self.shap_tabular_focus_sample(data,
                                                                         classifier_f,
                                                                         label,
                                                                         n_samples,
                                                                         focus_sample_batch_size,
                                                                         focus_sample_initial_points,
                                                                         to_consider=to_consider,
                                                                         enumerate_initial=enumerate_initial)
                rdata = focused_sampling_output[SDATA]
                y = focused_sampling_output[SY]
                weights = self._get_shap_weights(rdata, rdata.shape[1])
                blr = BayesianLinearRegression(percent=self.cred_int, l2=l2)
                blr.fit(rdata, y, weights)
            else:
                # Use ptg to get the number of samples, then focus sample
                # Note, this case isn't used in the paper and currently isn't implemented
                raise NotImplementedError
        else:
            # Use PTG to get initial samples

            # Enumerate intial points if requested
            if enumerate_initial:
                data_init, inverse_init = self._enumerate_initial_shap(data, max_coefs)
                n_more = ptg_initial_points - inverse_init.shape[0]
            else:
                n_more = ptg_initial_points

            # Perturb using initial samples
            rdata, inverse = self._shap_tabular_perturb_n_samples(data, n_more, max_coefs)
            if enumerate_initial:
                rdata = np.concatenate((data_init, rdata), axis=0)
                inverse = np.concatenate((inverse_init, inverse), axis=0)

            # Get labels
            y = classifier_f(inverse)[:, label]

            # Fit BLR
            weights = self._get_shap_weights(rdata, M=rdata.shape[1])
            blr = BayesianLinearRegression(percent=self.cred_int, l2=l2)
            blr.fit(rdata, y, weights)

            # Compute PTG number needed
            n_needed = int(np.ceil(blr.get_ptg(cred_width)))
            ptg_rdata, ptg_inverse  = self._shap_tabular_perturb_n_samples(data, 
                                                                           n_needed - ptg_initial_points,
                                                                           max_coefs)

            if self.verbose:
                tqdm.write(f"{n_needed} more samples needed")

            rdata = np.concatenate((rdata, ptg_rdata), axis=0)
            inverse = np.concatenate((inverse, ptg_inverse), axis=0)
            ptgy = classifier_f(ptg_inverse)[:, label]
            weights = self._get_shap_weights(rdata, M=rdata.shape[1])

            # Run final model
            ptgy = classifier_f(ptg_inverse)[:, label]
            y = np.concatenate((y, ptgy), axis=0)
            blr = BayesianLinearRegression(percent=self.cred_int, l2=l2)
            blr.fit(rdata, y, weights)    

        # Format output for returning
        output = {
            "data": rdata,
            "y": y,
            "distances": weights,
            "blr": blr,
            "coef": blr.coef_,
            "max_coefs": max_coefs
        }

        return output

    def explain(self,
                data,
                classifier_f,
                label,
                cred_width=1e-2,
                focus_sample=True,
                n_samples=None,
                max_n_samples=10_000,
                focus_sample_batch_size=2_500,
                focus_sample_initial_points=100,
                ptg_initial_points=200,
                to_consider=10_000,
                feature_selection=True,
                n_features=15,
                tag=None,
                only_coef=False,
                only_blr=False,
                enumerate_initial=True,
                max_coefs=None,
                l2=True):
        """Explain an instance.

        As opposed to other model agnostic explanations, the bayes explanations
        accept a credible interval width instead of a number of perturbations
        value.

        If the credible interval is set to 95% (as is the default), the bayesian
        explanations will generate feature importances that are +/- width/2
        95% of the time.


        Arguments:
            data: The data instance to explain
            classifier_f: The classification function. This function should return
                          probabilities for each label, where if there are M labels
                          and N instances, the output is of shape (N, M).
            label: The label index to explain.
            cred_width: The width of the credible interval of the resulting explanation. Note,
                   this serves as a upper bound in the implementation, the final credible
                   intervals may be tighter, because PTG is a bit approximate. Also, be 
                   aware that for kernelshap, if we can compute the kernelshap values exactly
                   by enumerating all the coalitions.
            focus_sample: Whether to use uncertainty sampling. 
            n_samples: If specified, n_samples with override the width setting feature
                       and compute the explanation with n_samples.
            max_n_samples: The maximum number of samples to use. If the width is set to 
                           a very small value and many samples are required, this serves
                           as a point to stop sampling.
            focus_sample_batch_size: The batch size of focus sampling.
            focus_sample_initial_points: The number of perturbations to collect before starting
                                         focused sampling.
            ptg_initial_points: The number perturbations to collect before computing the ptg estimate.
            to_consider: The number of perturbations to consider in focused sampling.
            feature_selection: Whether to do feature selection using Ridge regression. Note, currently
                               only implemented for BayesSHAP.
            n_features: The number of features to use in feature selection.
            tag: A tag to add the explanation.
            only_coef: Only return the explanation means.
            only_blr: Only return the bayesian regression object.
            enumerate_initial: Whether to enumerate a set of initial shap coalitions.
            l2: Whether to fit with l2 regression. Turning off the l2 regression can be useful for the shapley value estimation.
        Returns:
            explanation: The resulting feature importances, credible intervals, and bayes regression
                         object.
        """
        assert isinstance(data, np.ndarray), "Data must be numpy array. Note, this means that classifier_f \
                                              must accept numpy arrays."
        self._log_args(locals())

        if self.kernel == "lime" and self.data in ["tabular", "image"]:
            output = self._explain_bayes_lime(data, 
                                              classifier_f, 
                                              label, 
                                              focus_sample, 
                                              cred_width, 
                                              n_samples, 
                                              max_n_samples,
                                              focus_sample_batch_size,
                                              focus_sample_initial_points,
                                              ptg_initial_points,
                                              to_consider)
        elif self.kernel == "shap" and self.data in ["tabular", "image"]:
            output = self._explain_bayes_shap(data, 
                                              classifier_f, 
                                              label, 
                                              focus_sample, 
                                              cred_width, 
                                              n_samples, 
                                              max_n_samples,
                                              focus_sample_batch_size,
                                              focus_sample_initial_points,
                                              ptg_initial_points,
                                              to_consider,
                                              feature_selection=feature_selection,
                                              n_features=n_features,
                                              enumerate_initial=enumerate_initial,
                                              max_coefs=max_coefs,
                                              l2=l2)
        else: 
            pass

        output['tag'] = tag

        if only_coef:
            return output['coef']

        if only_blr:
            return output['blr']

        return output


def nCk(n, r):
    """n choose r

    From: https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python"""
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


def do_exp(args):
    """Supporting function for the explanations."""
    i, data, init_kwargs, exp_kwargs, labels, max_coefs, pass_args = args
    def do(data_i, label):

        if pass_args is not None and pass_args.balance_background_dataset:
            init_kwargs['training_data'] = np.concatenate((data_i[None, :], np.zeros((1, data_i.shape[0]))), axis=0)

        exp = BayesLocalExplanations(**init_kwargs)
        exp_kwargs['tag'] = i
        exp_kwargs['label'] = label
        if max_coefs is not None:
            exp_kwargs['max_coefs'] = max_coefs[i]
        e = deepcopy(exp.explain(data_i, **exp_kwargs))
        return e
    if labels is not None:
        return do(data[i], labels[i])
    else:
        return do(data[i], exp_kwargs['label'])


def explain_many(all_data, init_kwargs, exp_kwargs, pool_size=1, verbose=False, labels=None, max_coefs=None, args=None):
    """Parallel explanations."""
    with Pool(pool_size) as p:
        if verbose:
            results = list(tqdm(p.imap(do_exp, [(i, all_data, init_kwargs, exp_kwargs, labels, max_coefs, args) for i in range(all_data.shape[0])])))
        else:
            results = p.map(do_exp, [(i, all_data, init_kwargs, exp_kwargs, labels, max_coefs, args) for i in range(all_data.shape[0])])
        return results
