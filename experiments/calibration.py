"""Calibration experiments."""
import argparse
import os
from os.path import exists, dirname
import logging
import warnings
import sys

from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import tqdm

import lime.lime_tabular as baseline_lime_tabular
import shap

# Make sure we can get bayes explanations
parent_dir = dirname(os.path.abspath(os.getcwd()))
sys.path.append(parent_dir)

from bayes.explanations import BayesLocalExplanations, explain_many
from bayes.data_routines import get_dataset_by_name
from bayes.models import *


parser = argparse.ArgumentParser()
parser.add_argument("--kernel", required=True, help="The kernel, i.e., lime or shap.")
parser.add_argument("--dataset", required=True, help="The dataset to run on.")
parser.add_argument("--n_initial", default=100, type=int, help="The intial points to compute the calibration.")
parser.add_argument("--n_true", default=10_000, type=int, help="The amount of perturbations to compute the converged explanation.")
parser.add_argument("--n_threads", default=1, type=int, help="The number of threads to launch during the experiment.")
parser.add_argument("--num", type=int, default=None, help="The number of instances to run on. Leave set to None to run on all test instances.")
parser.add_argument("--verbose", action="store_true", help="Verbose output.")
parser.add_argument("--balance_background_dataset", action="store_true", help="Whether to balance the background sampling. This helps with tabular calibration.")
parser.add_argument("--seed", default=0, type=int)


def get_creds(initial, final, total_init=0.0, inside_init=0.0):
    """Computes the calibration from the initial and psuedo ground truth feature importances."""
    total, inside = total_init, inside_init
    for q, item in tqdm.tqdm(enumerate(initial)):
        creds = item.creds
        init_coef = item.coef_
        for i, c in enumerate(item.coef_):
            total += 1.0
            if final[q][i] <= (init_coef[i] + creds[i]) and final[q][i] >= (init_coef[i] - creds[i]):
                inside += 1
    return inside / total, total, inside


def run_calibration(args):
    """Runs the calibration experiment."""

    # Get data and model
    data = get_dataset_by_name(args.dataset)
   
    if args.dataset in ["compas", "german"]:
        image_dataset = False
        model_and_data = process_tabular_data_get_model(data)
    elif args.dataset[:5] in ["mnist"]:
        image_dataset = True
        model_and_data = process_mnist_get_model(data)
    elif args.dataset[:8] == "imagenet":
        image_dataset = True
        model_and_data = process_imagenet_get_model(data)
    else:
        raise NotImplementedError

    if image_dataset:
        xtest = model_and_data["xtest"]
        ytest = model_and_data["ytest"]
        segs = model_and_data["xtest_segs"]
        get_model = model_and_data["model"]
        label = model_and_data["label"]

        if args.num is None:
            args.num = xtest.shape[0]

        total, inside = 0.0, 0.0
        for i in tqdm.tqdm(range(args.num)):
            instance = xtest[i]
            segments = segs[i]
            cur_model = get_model(instance, segments)
            xtrain = get_xtrain(segments)

            # Get initial
            exp_init = BayesLocalExplanations(training_data=xtrain,
                                              data="image",
                                              kernel=args.kernel,
                                              categorical_features=np.arange(xtrain.shape[1]),
                                              verbose=args.verbose)
            rout = exp_init.explain(classifier_f=cur_model,
                                   data=np.ones_like(xtrain[0]),
                                   label=ytest[i],
                                   n_samples=args.n_initial,
                                   focus_sample=False,
                                   only_coef=False,
                                   only_blr=False)

            out = rout['blr']
            max_coef = rout['max_coefs']

            # Get 'ground truth'
            exp_final = BayesLocalExplanations(training_data=xtrain,
                                              data="image",
                                              kernel=args.kernel,
                                              categorical_features=np.arange(xtrain.shape[1]),
                                              verbose=args.verbose)
            out_final = exp_init.explain(classifier_f=cur_model,
                                         data=np.ones_like(xtrain[0]),
                                         label=ytest[i],
                                         n_samples=args.n_true,
                                         focus_sample=False,
                                         only_coef=True,
                                         max_coefs=max_coef)
   
            out = [out]
            out_final = [out_final]

            pct, total, inside = get_creds(out, out_final, total, inside)
            tqdm.tqdm.write(f"Calibration over {i+1} instances is {np.round(pct, 4)}")
        print(f"Final Calibration {pct} for {args.kernel} {args.dataset}")
    else:
        # For table datasets, we use explain_many here to allow parallel runs through n_threads
        xtrain = model_and_data["xtrain"]
        xtest = model_and_data["xtest"]
        model = model_and_data["model"]
        feature_names = data["column_names"]
        categorical_indices = data["cat_indices"]
        label = model_and_data["label"]

        if args.num is None:
            args.num = xtest.shape[0]

        print(f"Running calibration for {args.num} test instances...")
        
        init_kwargs = {
            "training_data": xtrain,
            "data": "tabular",
            "kernel": args.kernel,
            "discretize_continuous": True,
            "verbose": True,
            "categorical_features": categorical_indices,
        }
        exp_kwargs = {
            "classifier_f": model.predict_proba,
            "label": label,
            "n_samples": args.n_initial,
            "focus_sample": False,
            "only_coef": False,
            "only_blr": False
        } 

        labels = model.predict(xtest[:args.num])
        initial = explain_many(xtest[:args.num], init_kwargs, exp_kwargs, pool_size=args.n_threads, verbose=args.verbose, labels=labels, args=args)
        max_coefs = [explanation['max_coefs'] for explanation in initial]
        init_blrs = [explanation['blr'] for explanation in initial]

        exp_kwargs = {
            "classifier_f": model.predict_proba,
            "label": label,
            "n_samples": args.n_true,
            "focus_sample": False,
            "only_coef": True,
            "only_blr": False
        }   

        labels = model.predict(xtest[:args.num])
        final = explain_many(xtest[:args.num], init_kwargs, exp_kwargs, pool_size=args.n_threads, verbose=args.verbose, labels=labels, max_coefs=max_coefs, args=args)
        pct_included = get_creds(init_blrs, final)
        print(f"Final Calibration: {pct_included} for {args.kernel} {args.dataset}")

if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    run_calibration(args)