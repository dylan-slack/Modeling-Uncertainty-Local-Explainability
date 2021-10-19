"""Routines for processing data."""
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage.segmentation import slic, mark_boundaries

import torch
from torchvision import datasets, transforms

# The number of segments to use for the images
NSEGMENTS = 20
PARAMS = {
    'protected_class': 1,
    'unprotected_class': 0,
    'positive_outcome': 1,
    'negative_outcome': 0
}
IMAGENET_LABELS = {
    'french_bulldog': 245,
    'scuba_diver': 983,
    'corn': 987,
    'broccoli': 927
}

def get_and_preprocess_compas_data():
    """Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis
    
    Parameters
    ----------
    params : Params
    Returns
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    PROTECTED_CLASS = PARAMS['protected_class']
    UNPROTECTED_CLASS = PARAMS['unprotected_class']
    POSITIVE_OUTCOME = PARAMS['positive_outcome']
    NEGATIVE_OUTCOME = PARAMS['negative_outcome']

    compas_df = pd.read_csv("../data/compas-scores-two-years.csv", index_col=0)
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]

    compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
    X = compas_df[['age', 'two_year_recid','c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]

    # if person has high score give them the _negative_ model outcome
    y = np.array([NEGATIVE_OUTCOME if score == 'High' else POSITIVE_OUTCOME for score in compas_df['score_text']])
    sens = X.pop('race')

    # assign African-American as the protected class
    X = pd.get_dummies(X)
    sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
    X['race'] = sensitive_attr

    # make sure everything is lining up
    assert all((sens == 'African-American') == (X['race'] == PROTECTED_CLASS))
    cols = [col for col in X]

    categorical_features = [1, 4, 5, 6, 7, 8]

    output = {
        "X": X.values,
        "y": y,
        "column_names": cols,
        "cat_indices": categorical_features
    }
    
    return output

def get_and_preprocess_german():
    """"Handle processing of German.  We use a preprocessed version of German from Ustun et. al.
    https://arxiv.org/abs/1809.06514.  Thanks Berk!
    Parameters:
    ----------
    params : Params
    Returns:
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    PROTECTED_CLASS = PARAMS['protected_class']
    UNPROTECTED_CLASS = PARAMS['unprotected_class']
    POSITIVE_OUTCOME = PARAMS['positive_outcome']
    NEGATIVE_OUTCOME = PARAMS['negative_outcome']

    X = pd.read_csv("../data/german_processed.csv")
    y = X["GoodCustomer"]

    X = X.drop(["GoodCustomer", "PurposeOfLoan"], axis=1)
    X['Gender'] = [1 if v == "Male" else 0 for v in X['Gender'].values]

    y = np.array([POSITIVE_OUTCOME if p == 1 else NEGATIVE_OUTCOME for p in y.values])
    categorical_features = [0, 1, 2] + list(range(9, X.shape[1]))

    output = {
        "X": X.values,
        "y": y,
        "column_names": [c for c in X],
        "cat_indices": categorical_features,
    }

    return output

def get_PIL_transf(): 
    """Gets the PIL image transformation."""
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    
    return transf  

def load_image(path):
    """Loads an image by path."""
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

def get_imagenet(name, get_label=True):
    """Gets the imagenet data.

    Arguments:
        name: The name of the imagenet dataset
    """
    images_paths = []

    # Store all the paths of the images
    data_dir = os.path.join("../data", name)
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for fn in filenames:
            if fn != ".DS_Store":
                images_paths.append(os.path.join(dirpath, fn))
    
    # Load & do transforms for the images
    pill_transf = get_PIL_transf()
    images, segs = [], []
    for img_path in images_paths:
        img = load_image(img_path)
        PIL_transformed_image = np.array(pill_transf(img))
        segments = slic(PIL_transformed_image, n_segments=NSEGMENTS, compactness=100, sigma=1)

        images.append(PIL_transformed_image)
        segs.append(segments)

    images = np.array(images)

    if get_label:
        assert name in IMAGENET_LABELS, "Get label set to True but name not in known imagenet labels"
        y = np.ones(images.shape[0]) * IMAGENET_LABELS[name]
    else:
        y = np.ones(images.shape[0]) * -1

    segs = np.array(segs)

    output = {
        "X": images,
        "y": y,
        "segments": segs
    }

    return output


def get_mnist(num):
    """Gets the MNIST data for a certain digit.

    Arguments: 
        num: The mnist digit to get
    """

    # Get the mnist data
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/mnist', 
                                                             train=False, 
                                                             download=True, 
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                            transforms.Normalize((0.1307,), (0.3081,))
                                                                                           ])),
                                                             batch_size=1, 
                                                             shuffle=False)

    all_test_mnist_of_label_num, all_test_segments_of_label_num = [], []

    # Get all instances of label num
    for data, y in test_loader:
        if y[0] == num:
            # Apply segmentation
            sample = np.squeeze(data.numpy().astype('double'),axis=0)
            segments = slic(sample.reshape(28,28,1), n_segments=NSEGMENTS, compactness=1, sigma=0.1).reshape(1,28,28)
            all_test_mnist_of_label_num.append(sample)
            all_test_segments_of_label_num.append(segments)

    all_test_mnist_of_label_num = np.array(all_test_mnist_of_label_num)
    all_test_segments_of_label_num = np.array(all_test_segments_of_label_num)

    output = {
        "X": all_test_mnist_of_label_num,
        "y": np.ones(all_test_mnist_of_label_num.shape[0]) * num,
        "segments": all_test_segments_of_label_num
    }

    return output

def get_dataset_by_name(name, get_label=True):
    if name == "compas":
        d = get_and_preprocess_compas_data()
    elif name == "german":
        d = get_and_preprocess_german()
    elif "mnist" in name:
        d = get_mnist(int(name[-1]))
    elif "imagenet" in name:
        d = get_imagenet(name[9:], get_label=get_label)
    else:
        raise NameError("Unkown dataset %s", name)
    d['name'] = name
    return d
