"""Stability experiments."""
import argparse
from copy import deepcopy
import logging
import os
from os.path import exists, dirname
import sys
import warnings

import numpy as np
import pickle as pkl
import tqdm

import lime.lime_tabular as baseline_lime_tabular
import shap

parent_dir = dirname(os.path.abspath(os.getcwd()))
sys.path.append(parent_dir)

from bayes.explanations import BayesLocalExplanations, explain_many
from bayes.data_routines import get_dataset_by_name
from bayes.models import process_tabular_data_get_model, process_mnist_get_model, get_xtrain, process_imagenet_get_model

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--kernel", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--save_loc", required=True)
parser.add_argument("--n_examples", required=True, type=int, default=10)
parser.add_argument("--n_threads", type=int, default=2, help="Number of threads to start while generating explanations.")
parser.add_argument("--n_samples", default=5_000, type=int)
parser.add_argument("--batch_size", default=2_500, type=int)
parser.add_argument("--overwrite_save", default=False, type=bool)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--seed", default=0, type=int)


def get_epsilon_tightness(args):
    """Gets the epsilon tightness for stability experiments."""
    if args.dataset == "compas":
        eps = 0.1
    elif args.dataset == "german":
        eps = 0.1
    else:
        raise NotImplementedError
    return eps


def get_all_points_less_than_eps(epsilon, X, n_examples=100):
    """Gets all points in the data closer than l2 distance of epsilon."""
    assert n_examples <= X.shape[0], f"n_examples is {n_examples} but data only contains {X.shape[0]} instances." 

    neighbors = []
    pbar = tqdm.tqdm(total=n_examples)
    i = 0

    while len(neighbors) < n_examples:
        c_epsilon = epsilon
        c_x = X[i]
        c_neighbors = []
        while len(c_neighbors) == 0:
            for j in range(X.shape[0]):
                if i == j or all(c_x == X[j]):
                    continue
                if np.linalg.norm(c_x - X[j]) < c_epsilon:
                    c_neighbors.append(X[j])
            # If we can't find neighbors in radius, increment
            # melis leaves this at 0.1 it seems like but
            # it doesn't always includes enough points, so increment
            # until we get at least one
            c_epsilon += 0.1
        neighbors.append(c_neighbors)

        if i == X.shape[0]:
            raise NameError("Couldn't find enough points with neighbors")

        i += 1
        pbar.update(1)

    return neighbors


def map_to_list(exp, label):
    exp = exp.local_exp[label]
    return np.array([item[1] for item in exp])


def get_baseline_explanation(model, instance, args, X, feature_names, categorical_indices, label):
    """Gets the either lime or shap baseline."""
    if args.kernel == "shap":
        # Make sure that discretization is the same as in bayesshap
        get_discretizer = BayesLocalExplanations(X, 
                                                 data="tabular", 
                                                 kernel=args.kernel, 
                                                 discretize_continuous=True, 
                                                 verbose=False, 
                                                 categorical_features=categorical_indices)
        discretized_instance = get_discretizer.shap_info.discretizer.discretize(instance)

        # Prediction function
        def predict(arr):
            substituted_instances = deepcopy(arr)
            for i in range(substituted_instances.shape[0]):
                substituted_instances[i, substituted_instances[i] == 1] = discretized_instance[0, substituted_instances[i] == 1]
            final = get_discretizer.shap_info.discretizer.undiscretize(substituted_instances)
            return model(final)

        explainer = shap.KernelExplainer(predict, np.zeros_like(discretized_instance))
        exp = explainer.shap_values(discretized_instance, n_samples=args.n_samples, l1_reg="num_features({})".format(X.shape[1]))
        original_mean = exp[label][0]
    else:
        explainer = baseline_lime_tabular.LimeTabularExplainer(X, 
                                                               discretize_continuous=True,
                                                               categorical_features=categorical_indices)
        exp = explainer.explain_instance(instance[0], 
                                         model, 
                                         num_samples=args.n_samples,
                                         num_features=X.shape[1], 
                                         labels=(label,))
        original_mean = map_to_list(exp, label)
    return original_mean


def add_image_epsilon(X, n_examples=100, mnist=False):
    """Performs the epsilon permutation for n_examples number of images."""
    X_to_eps = X[:n_examples]

    if mnist:
        n_p = np.random.normal(loc=0, scale=1e-2, size=(n_examples, 5, 1, 28, 28))
    else:
        n_p = np.random.normal(loc=0, scale=1, size=(n_examples, 5, 224, 224, 3))

    # Perturb instances
    neighbors = []
    for i in range(n_examples):
        neighbors.append(X_to_eps[i] + n_p[i])
    neighbors = np.array(neighbors)
    
    return neighbors


def calculate_lip(neighbor_means, neighbors_points, original_mean, original_point):
    max_lip = 0
    for m, p in zip(neighbor_means, neighbors_points):
        lip = np.linalg.norm(original_mean - m) / np.linalg.norm(original_point - p)
        if lip > 1e30:
            warnings.warn("LIP overflow")
            continue
        max_lip = max(lip, max_lip)
    return max_lip


def run_stability(args):
    """Runs the stability experiment."""
    if not args.overwrite_save:
        assert not exists(args.save_loc), f"Save location {args.save_loc} already has data"

    # Get data and model
    data = get_dataset_by_name(args.dataset)

    if args.dataset in ["compas", "german"]:
        image_dataset = False
        model_and_data = process_tabular_data_get_model(data)
    elif args.dataset[:5] in ["mnist"]:
        image_dataset = True
        mnist = True 
        model_and_data = process_mnist_get_model(data)
    elif args.dataset[:8] == "imagenet":
        image_dataset = True
        mnist = False
        model_and_data = process_imagenet_get_model(data)
    else:
        raise NotImplementedError

    if image_dataset:
        
        xtest = model_and_data["xtest"]
        ytest = model_and_data["ytest"]
        segs = model_and_data["xtest_segs"]
        get_model = model_and_data["model"]
        label = model_and_data["label"]

        all_neighborhoods = add_image_epsilon(xtest, n_examples=args.n_examples, mnist=mnist)
        
        ########### Baseline LIP ###########
        logging.info(f"Running {args.kernel} stability baselines on {args.dataset}")
        baseline_max_lips = []

        for i in tqdm.tqdm(range(args.n_examples)):

            # Wrap model for current instance and segments
            instance = xtest[i]
            segments = segs[i]
            xtrain = get_xtrain(segments)
            model = get_model(instance, segments)

            # Get baseline explanation on original instance
            original_mean = get_baseline_explanation(model=model, 
                                                     instance=np.ones_like(xtrain[:1]),
                                                     args=args,
                                                     X=xtrain,
                                                     feature_names=None,
                                                     categorical_indices=np.arange(xtrain.shape[1]),
                                                     label=ytest[i])
            # Get explanation on all neighbors
            neighbor_means = []
            neighbors = []
            for q in range(len(all_neighborhoods[i])):
                instance = all_neighborhoods[i][q]
                segments = segs[i]
                neighbor_model = get_model(instance, segments)
                neighbor_mean = get_baseline_explanation(model=neighbor_model, 
                                                         instance=np.ones_like(xtrain[:1]),
                                                         args=args,
                                                         X=xtrain,
                                                         feature_names=None,
                                                         categorical_indices=np.arange(xtrain.shape[1]),
                                                         label=ytest[i])
                neighbors.append(all_neighborhoods[i][q])
                neighbor_means.append(neighbor_mean)
            baseline_max_lips.append(calculate_lip(neighbor_means, neighbors, original_mean, xtest[i]))
        print(baseline_max_lips)
        bayes_max_lips = []
        ########### Bayes LIP ###########
        for i in tqdm.tqdm(range(args.n_examples)):
            instance = xtest[i]
            segments = segs[i]
            model = get_model(instance, segments)
            exp_init = BayesLocalExplanations(training_data=xtrain,
                                              data="image",
                                              kernel=args.kernel,
                                              categorical_features=np.arange(xtrain.shape[1]),
                                              verbose=args.verbose)
            out = exp_init.explain(classifier_f=model,
                                   data=np.ones_like(xtrain[0]),
                                   label=ytest[i],
                                   n_samples=args.n_samples,
                                   focus_sample=True,
                                   focus_sample_batch_size=args.batch_size,
                                   feature_selection=False)
            bayes_original_mean = out['blr'].coef_

            bayes_neighbor_means = []
            bayes_neighbors = []
            for q in range(len(all_neighborhoods[i])):
                instance = all_neighborhoods[i][q]
                neighbor_model = get_model(instance, segments)
                out = exp_init.explain(classifier_f=neighbor_model,
                                   data=np.ones_like(xtrain[0]),
                                   label=ytest[i],
                                   n_samples=args.n_samples,
                                   focus_sample=True,
                                   focus_sample_batch_size=args.batch_size,
                                   feature_selection=False)
                bayes_neighbors.append(all_neighborhoods[i][q])
                bayes_neighbor_means.append(out['blr'].coef_)
            bayes_max_lips.append(calculate_lip(bayes_neighbor_means, bayes_neighbors, bayes_original_mean, xtest[i]))
            print(bayes_max_lips)
    else:
        epsilon = get_epsilon_tightness(args)

        xtrain = model_and_data["xtrain"]
        xtest = model_and_data["xtest"]
        model = model_and_data["model"]
        feature_names = data["column_names"]
        categorical_indices = data["cat_indices"]
        label = model_and_data["label"]

        # Get neighbors
        all_neighborhoods = get_all_points_less_than_eps(epsilon, xtest, n_examples=args.n_examples)

        ########### Baseline LIP ###########
        logging.info(f"Running {args.kernel} stability baselines on {args.dataset}")
        baseline_max_lips = []
        for i in tqdm.tqdm(range(args.n_examples)):
            original_mean = get_baseline_explanation(model=model.predict_proba, 
                                                     instance=xtest[i:i+1],
                                                     args=args,
                                                     X=xtrain,
                                                     feature_names=feature_names,
                                                     categorical_indices=categorical_indices,
                                                     label=label)
            neighbor_means = []
            neighbors = []
            for q in range(len(all_neighborhoods[i])):
                neighbor_mean = get_baseline_explanation(model=model.predict_proba, 
                                                         instance=all_neighborhoods[i][q].reshape(1, -1),
                                                         args=args,
                                                         X=xtrain,
                                                         feature_names=feature_names,
                                                         categorical_indices=categorical_indices,
                                                         label=label)

                    
                neighbors.append(all_neighborhoods[i][q])
                neighbor_means.append(neighbor_mean)
            baseline_max_lips.append(calculate_lip(neighbor_means, neighbors, original_mean, xtest[i]))
        print(baseline_max_lips)
        ###################################

        ########### Focused LIP ###########
        logging.info(f"Running {args.kernel} stability focused sampling on {args.dataset}")
        bayes_max_lips = []
        for i in tqdm.tqdm(range(args.n_examples)):
            neighbor_means = []
            neighbors = []

            init_kwargs = {
                "training_data": xtrain,
                "data": "tabular",
                "kernel": args.kernel,
                "discretize_continuous": True,
                "verbose": True,
                "categorical_features": categorical_indices
            }

            exp_kwargs = {
                "classifier_f": model.predict_proba,
                "label": label,
                "n_samples": args.n_samples,
                "focus_sample": True,
                "only_coef": True,
                "feature_selection": False,
                "focus_sample_batch_size": args.batch_size,
                "enumerate_initial": True,
            }   

            data_to_explain = np.concatenate((xtest[i].reshape(1, -1), np.array(all_neighborhoods[i])), axis=0)
            output = explain_many(data_to_explain, init_kwargs, exp_kwargs, pool_size=args.n_threads)
            original_mean = output[0]
            neighbor_means = output[1:]
            bayes_max_lips.append(calculate_lip(neighbor_means, all_neighborhoods[i], original_mean, xtest[i]))
            print(bayes_max_lips)

    with open(args.save_loc, "wb") as f:
        pkl.dump([baseline_max_lips, bayes_max_lips], f)


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    run_stability(args)

