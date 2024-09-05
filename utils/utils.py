import os.path
import pickle

from taxonomy.utils import available_aps
from pathlib import Path
from taxonomy.utils import get_gdrive_id
import torch
import gdown
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def download_gdrive_new(gdrive_id, fname_save, saved_model_path):
    """Robustbench implementation from https://github.com/wkentaro/gdown."""

    # check if file was already downloaded
    if not os.path.isfile(saved_model_path):
        # check if the apb_pruned_models dir has been created
        if not os.path.isdir(fname_save):
            # if dir was never been created
            os.makedirs(str(fname_save))
            gdown.download(id=gdrive_id, output=fname_save)
        else:
            # if dir was created and file was not there
            gdown.download(id=gdrive_id, output=fname_save)

    else:
        print('AP File already downloaded and verified')


def model_key_maker(ap_method=None, architecture='resnet18', dataset='CIFAR10', structure='weights', sparsity_rate='90'):
    """
    This method produces the key to be uniformly used to download the gdrive file.
    Also, it serves as a sanity check for the input entries.
    """

    # sanity check
    if not available_aps(ap_method):
        raise KeyError(
            'The select AP method does not exist: please, check the available keys on the taxonomy or update the '
            'benchmark to the latest version.')
    if architecture not in ['resnet18', 'vgg16']:
        raise KeyError('Choose your architecture between resnet18 or vgg16.')
    if dataset not in ['CIFAR10', 'SVHN']:
        raise KeyError('Choose your dataset between CIFAR10 or SVHN')
    if structure not in ['weights', 'filters', 'channels']:
        raise KeyError('Choose between weights, channels or filters pruning (note: use the plural of the parameter).')
    if (structure == 'filter' or structure == 'channel') and (sparsity_rate not in ["50", "75", "90"]):
        raise KeyError('Unavailable sparsity for structured pruning: select, with string type, among 50, 75, 90.')
    if structure == 'weight' and sparsity_rate not in ["90", "95", "99"]:
        raise KeyError('Unavailable sparsity for structured pruning: select, with string type, among 90, 95, 99.')

    # get architecture
    ar = "R18" if architecture == 'resnet18' else "V16"
    st = "US" if structure == 'weights' else "S"

    # return the key
    return ap_method + '_' + ar + '_' + dataset + '_' + st + '_' + sparsity_rate


def load_model(model_key=None, normalization=False):
    """
    This method aims to
    :param model_key: The model key used in google drive.
    :param get_distances: If true, loads the distances from gdrive and returns them.
    :return: model, distances (optional)
    """

    # test load in base model
    if 'base' in model_key:
        model = models.__dict__(model_key)
        model(normalization=normalization)

    else:
        model_gdrive_id, ext = get_gdrive_id(model_key, kind='model')
        model_path = './ap_models/'
        saved_model_path = model_path + model_key + ext
        # download the checkpoint
        download_gdrive_new(model_gdrive_id, model_path, saved_model_path)
        # define saved model path
        checkpoint = torch.load(saved_model_path, map_location=device)

        # get ap_method+arch to load model
        third_idx = model_key.find('_', model_key.find('_', model_key.find('_') + 1) + 1)
        ap_arch = model_key[:third_idx]

        # load model from models
        model = models.__dict__[ap_arch]()

        # load checkpoint
        if 'FlyingBird' in ap_arch:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=True)
            except KeyError:
                pass
            try:
                model.load_state_dict(checkpoint['net'], strict=True)
            except KeyError:
                pass

    return model


def load_distance(model_key=None):
    """
    This method aims to
    :param model_key: The model key used in google drive.
    :param get_distances: If true, loads the distances from gdrive and returns them.
    :return: model, distances (optional)
    """

    distance_gdrive_id, ext = get_gdrive_id(model_key, kind='distance')
    distance_path = './ap_distances/'
    saved_distance_path = distance_path + model_key + ext
    # download the checkpoint
    download_gdrive_new(distance_gdrive_id, distance_path, saved_distance_path)
    with open(saved_distance_path, 'rb') as handle:
        distance = pickle.load(handle)

    return distance
