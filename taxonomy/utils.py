from taxonomy.models import models_gdrives
from taxonomy.distances import distances_gdrives


def get_gdrive_id(model_key, kind='model'):
    if kind == 'model':
        return models_gdrives[model_key][0], models_gdrives[model_key][1]
    elif kind == 'distance':
        return distances_gdrives[model_key][0], distances_gdrives[model_key][1]
    else:
        raise KeyError('Choose between model and distance kind.')


def available_aps(model_key):
    """
    Checks if there's any of the given AP available for download
    """
    # take dimension of input key
    key_len = len(model_key)
    # take all keys
    cut_keys = [item[0:key_len] for item in list(models_gdrives.keys())]
    return model_key in cut_keys


import json


def load_ap_taxonomy(ap):
    # cut the ap key to only the relevant part
    ap_prefix = "_".join(ap.split("_")[:2])

    # load the taxonomy from the JSON file
    with open('taxonomy/taxonomy.json', 'r') as f:
        taxonomy = json.load(f)

    # find and return the matching entry
    for entry in taxonomy:
        if entry.get('ap_key') == ap_prefix:
            return entry

    # if no matching entry is found, return None or raise an error
    return None

