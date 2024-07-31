import os.path


from taxonomy.utils import available_aps
from pathlib import Path
from taxonomy.utils import get_gdrive_id
import torch
import gdown
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def download_gdrive_new(gdrive_id, fname_save):
    """Robustbench implementation from https://github.com/wkentaro/gdown."""

    if not os.path.isfile(fname_save):
        if isinstance(fname_save, Path):
            fname_save = str(fname_save)
        print(f'Downloading {fname_save} (gdrive_id={gdrive_id}).')
        gdown.download(id=gdrive_id, output=fname_save)
    else:
        print('Model already downloaded and verified.')


def model_key_maker(ap_method=None, architecture='resnet18', dataset='CIFAR10', structure=struct, sparsity_rate=sr):
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
    if structure not in ['weight', 'filter', 'channel']:
        raise KeyError('Choose between weight or filter pruning.')
    if (structure == 'filter' or structure == 'channel') and (sparsity_rate not in ["50", "75", "90"]):
        raise KeyError('Unavailable sparsity for structured pruning: select, with string type, among 50, 75, 90.')
    if structure == 'weight' and sparsity_rate not in ["90", "95", "99"]:
        raise KeyError('Unavailable sparsity for structured pruning: select, with string type, among 90, 95, 99.')

    # get architecture
    ar = "R18" if architecture == 'resnet18' else "V16"
    st = "US" if structure == 'weight' else "S"

    # return the key
    return ap_method + '_' + ar + '_' + dataset + '_' + st + '_' + sparsity_rate


def load_model(model_key=None):
    """
    This method aims to
    :param model_key: The model key used in google drive.
    :param get_distances: If true, loads the distances from gdrive and returns them.
    :return: model, distances (optional)
    """
    model_gdrive_id, ext = get_gdrive_id(model_key)
    model_path = './apb_pruned_models/' + model_key+ext
    # download the checkpoint
    download_gdrive_new(model_gdrive_id, model_path)
    checkpoint = torch.load(model_path, map_location=device)

    # get ap_method+arch to load model
    third_idx = model_key.find('_', model_key.find('_', model_key.find('_') + 1) + 1)
    ap_arch = model_key[:third_idx]

    # load model from models
    model = models.__dict__[ap_arch]()
    # load checkpoint
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        pass
    try:
        model.load_state_dict(checkpoint['net'])
    except KeyError:
        pass

    return model
