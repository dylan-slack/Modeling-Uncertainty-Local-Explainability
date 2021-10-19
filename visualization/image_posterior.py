"""Create a gif sampling from the posterior from an image.

The file includes routines to create gifs of posterior samples for image
explanations. To create the gif, we sample a number of draws from the posterior,
plot the explanation and the image, and repeat this to stitch together a gif.

The interpretation is that regions of the image that more frequency show up as
green are more likely to positively impact the prediction. Similarly, regions that 
more frequently show up as red are more likey to negatively impact the prediction.
"""
import os
from os.path import exists, dirname
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
import tempfile
from tqdm import tqdm

import lime.lime_tabular as baseline_lime_tabular
import shap

# Make sure we can get bayes explanations
parent_dir = dirname(os.path.abspath(os.getcwd()))
sys.path.append(parent_dir)

from bayes.explanations import BayesLocalExplanations, explain_many
from bayes.data_routines import get_dataset_by_name
from bayes.models import *

def fill_segmentation(values, segmentation, image, n_max=5):
    max_segs = np.argsort(abs(values))[-n_max:]
    out = np.zeros((224, 224))
    c_image = np.zeros(image.shape)
    for i in range(len(values)):
        if i in max_segs:
            out[segmentation == i] = 1 if values[i] > 0 else -1
            c = 1 if values[i] > 0 else 0
            c_image[segmentation == i, c] = np.max(image)
    return c_image.astype(int), out.astype(int)

def create_gif(explanation_blr, segments, image, save_loc, n_images=20, n_max=5):
    """Create the gif corresponding to the image explanation.

    Arguments:
        explanation_coefficients: The explanation blr object.
        segments: The image segmentation.
        image: The image for which to compute the explantion.
        save_loc: The location to save the gif.
        n_images: Number of images to create the gif with.
        n_max: The number of superpixels to draw on the image.
    """
    draws = explanation_blr.draw_posterior_samples(n_images)
    # Setup temporary directory to store paths in 
    with tempfile.TemporaryDirectory() as tmpdirname:
        paths = []
        for i, d in tqdm(enumerate(draws)):
            c_image, filled_segs = fill_segmentation(d, segments, image, n_max=n_max)
            plt.cla()
            plt.axis('off')
            plt.imshow(mark_boundaries(image, filled_segs))
            plt.imshow(c_image, alpha=0.3)
            paths.append(os.path.join(tmpdirname, f"{i}.png"))
            plt.savefig(paths[-1])

        # Save to gif
        # https://stackoverflow.com/questions/61716066/creating-an-animation-out-of-matplotlib-pngs
        print(f"Saving gif to {save_loc}")
        ims = [imageio.imread(f) for f in paths]
        imageio.mimwrite(save_loc, ims)

