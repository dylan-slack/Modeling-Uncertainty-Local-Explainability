"""Routines that implement processing data & getting models.

This file includes various routines for processing & acquiring models, for
later use in the code. The table data preprocessing is straightforward. We
first applying scaling to the data and fit a random forest classifier.

The processing of the image data is a bit more complex. To simplify the construction
of the explanations, the explanations don't accept images. Instead, for image explanations,
it is necessary to define a function that accept a array of 0's and 1's corresponding to
segments for a particular image being either excluded or included respectively. The explanation
is performed on this array.
"""
import numpy as np
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torchvision import models, transforms

from data.mnist.mnist_model import Net

def get_xtrain(segs):
    """A function to get the mock training data to use in the image explanations.

    This function returns a dataset containing a single instance of ones and 
    another of zeros to represent the training data for the explanation. The idea
    is that the explanation will use this data to compute the perturbations, which
    will then be fed into the wrapped model.

    Arguments:
        segs: The current segments array
    """
    n_segs = len(np.unique(segs))
    xtrain = np.concatenate((np.ones((1, n_segs)), np.zeros((1, n_segs))), axis=0)
    return xtrain

def process_imagenet_get_model(data):
    """Gets wrapped imagenet model."""

    # Get the vgg16 model, used in the experiments
    model = models.vgg16(pretrained=True)
    model.eval()
    model.cuda()

    xtest = data['X']
    ytest = data['y'].astype(int)
    xtest_segs = data['segments']

    softmax = torch.nn.Softmax(dim=1)

    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])   

    t_xtest = transf(xtest[0])[None, :]#.cuda()

    # Define the wrapped model
    def get_wrapped_model(instance, segments, background=0, batch_size=64): 
        def wrapped_model(data):
            perturbed_images = []
            for d in data:
                perturbed_image = deepcopy(instance)
                for i, is_on in enumerate(d):
                    if is_on == 0:
                        perturbed_image[segments==i, 0] = background
                        perturbed_image[segments==i, 1] = background
                        perturbed_image[segments==i, 2] = background
                perturbed_images.append(transf(perturbed_image)[None, :])
            perturbed_images = torch.from_numpy(np.concatenate(perturbed_images, axis=0)).float().cuda()
            predictions = []
            for q in range(0, perturbed_images.shape[0], batch_size):
                predictions.append(softmax(model(perturbed_images[q:q+batch_size])).cpu().detach().numpy())
            predictions = np.concatenate(predictions, axis=0)
            return predictions
        return wrapped_model

    output = {
        "model": get_wrapped_model,
        "xtest": xtest,
        "ytest": ytest,
        "xtest_segs": xtest_segs,
        "label": data['y'][0]
    }

    return output

def process_mnist_get_model(data):
    """Gets wrapped mnist model."""
    xtest = data['X']
    ytest = data['y'].astype(int)
    xtest_segs = data['segments']

    model = Net()
    model.load_state_dict(torch.load("../data/mnist/mnist_cnn.pt"))
    model.eval()
    model.cuda()

    softmax = torch.nn.Softmax(dim=1)
    def get_wrapped_model(instance, segments, background=-0.4242, batch_size=100): 
        def wrapped_model(data):
            perturbed_images = []
            data = torch.from_numpy(data).float().cuda()
            for d in data:
                perturbed_image = deepcopy(instance)
                for i, is_on in enumerate(d):
                    if is_on == 0:
                        a = segments==i
                        perturbed_image[0, segments[0]==i] = background
                perturbed_images.append(perturbed_image[:, None])
            perturbed_images = torch.from_numpy(np.concatenate(perturbed_images, axis=0)).float().cuda()
            
            # Batch predictions if necessary
            if perturbed_images.shape[0] > batch_size:
                predictions = []
                for q in range(0, perturbed_images.shape[0], batch_size):
                    predictions.append(softmax(model(perturbed_images[q:q+batch_size])).cpu().detach().numpy())
                predictions = np.concatenate(predictions, axis=0)
            else:
                predictions = softmax(model(perturbed_images)).cpu().detach().numpy()
            return np.array(predictions)
        return wrapped_model

    output = {
        "model": get_wrapped_model,
        "xtest": xtest,
        "ytest": ytest,
        "xtest_segs": xtest_segs,
        "label": data['y'][0],
    }

    return output

def process_tabular_data_get_model(data):
    """Processes tabular data + trains random forest classifier."""
    X = data['X']
    y = data['y']

    xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)
    ss = StandardScaler().fit(xtrain)
    xtrain = ss.transform(xtrain)
    xtest = ss.transform(xtest)
    rf = RandomForestClassifier(n_estimators=100).fit(xtrain,ytrain)

    output = {
        "model": rf,
        "xtrain": xtrain,
        "xtest": xtest,
        "ytrain": ytrain,
        "ytest": ytest,
        "label": 1,
        "model_score": rf.score(xtest, ytest)
    }

    print(f"Model Score: {output['model_score']}")

    return output