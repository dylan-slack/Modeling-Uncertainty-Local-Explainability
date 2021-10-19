"""PTG experiments."""
import argparse
import os
from os.path import exists, dirname
import logging
import warnings
import sys

import numpy as np
import pickle as pkl
import tqdm

import lime.lime_tabular as baseline_lime_tabular
import shap

parent_dir = dirname(os.path.abspath(os.getcwd()))
sys.path.append(parent_dir)
 
from bayes.explanations import BayesLocalExplanations, explain_many
from bayes.data_routines import get_dataset_by_name
from bayes.models import process_tabular_data_get_model, process_mnist_get_model, get_xtrain

parser = argparse.ArgumentParser()
parser.add_argument("--kernel", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--n_initial", default=200, type=int)
parser.add_argument("--n_threads", default=1, type=int)
parser.add_argument("--num", type=int, default=None)
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--datatype", type=str, default="tabular")
parser.add_argument("--save_loc", type=str, default="results")
parser.add_argument("--widths", type=str, default="5e-3 6e-3 7e-3 8e-3 9e-3 1e-2")
parser.add_argument("--seed", default=0, type=int)


def run_ptg(args):
    """Runs the ptg experiment."""
    assert not exists(args.save_loc), f"Save location {args.save_loc} already has data"

    # Get data and model
    data = get_dataset_by_name(args.dataset)
   
    if args.dataset in ["compas", "german"]:
        image_dataset = False
        model_and_data = process_tabular_data_get_model(data)
    elif args.dataset[:5] in ["mnist"]:
        image_dataset = True
        model_and_data = process_mnist_get_model(data)
    else:
        raise NotImplementedError

    desired_widths = [float(w) for w in args.widths.split()]
    print(f"Using widths {desired_widths}")

    if image_dataset:

        # Get data
        xtest = model_and_data["xtest"]
        ytest = model_and_data["ytest"]
        segs = model_and_data["xtest_segs"]
        get_model = model_and_data["model"]
        label = model_and_data["label"]
        if args.num is None:
            args.num = xtest.shape[0]

        # Compute ptg for images
        results = []
        for w in tqdm.tqdm(desired_widths):
            tqdm.tqdm.write(f"Running with width {w}")
            cur_creds = []
            for i in tqdm.tqdm(range(args.num)):

                # Wrap the image model for current 
                # instance + segments
                instance = xtest[i]
                segments = segs[i]
                model = get_model(instance, segments)
                xtrain = get_xtrain(segments)

                # Compute the explanation
                exp_init = BayesLocalExplanations(training_data=xtrain,
                                                  data=args.datatype,
                                                  kernel=args.kernel,
                                                  categorical_features=np.arange(xtrain.shape[1]),
                                                  verbose=args.verbose)
                out = exp_init.explain(classifier_f=model,
                                       data=np.ones_like(xtrain[0]),
                                       label=ytest[0],
                                       cred_width=w,
                                       focus_sample=False,
                                       ptg_initial_points=args.n_initial)

                cur_creds.extend(out['blr'].creds)

            # Print out update for the current median observed widths
            c_median = np.round(np.median(cur_creds), 8)
            tqdm.tqdm.write(f"Median observed width is {c_median} for desired width {w}.")
            results.append(cur_creds)
    else:
        xtrain = model_and_data["xtrain"]
        xtest = model_and_data["xtest"]
        model = model_and_data["model"]
        categorical_indices = data["cat_indices"]
        label = model_and_data["label"]

        if args.num == None:
            args.num = xtest.shape[0]

        print(f"Running ptg for {args.num} test instances...")
        labels = model.predict(xtest[:args.num])
        results = []
        for w in tqdm.tqdm(desired_widths):
            tqdm.tqdm.write(f"Running with width {w}")
            cur_creds = []
            for i in tqdm.tqdm(range(args.num)):
                exp_init = BayesLocalExplanations(training_data=xtrain,
                                                  data=args.datatype,
                                                  kernel=args.kernel,
                                                  categorical_features=categorical_indices,
                                                  discretize_continuous=True,
                                                  verbose=args.verbose)
                out = exp_init.explain(classifier_f=model.predict_proba,
                                       data=xtest[i],
                                       label=labels[i],
                                       cred_width=w,
                                       focus_sample=False)
                cur_creds.extend(out['blr'].creds)
            results.append(cur_creds)

    with open(args.save_loc, "wb") as f:
        pkl.dump(results, f)


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(args.seed)
    run_ptg(args)
