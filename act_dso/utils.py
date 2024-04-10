"""Utility functions used in deep symbolic optimization."""

import collections
import copy
import os
import commentjson as json


def safe_merge_dicts(base_dict, update_dict):
    """Merges two dictionaries without changing the source dictionaries.

    Parameters
    ----------
        base_dict : dict
            Source dictionary with initial values.
        update_dict : dict
            Dictionary with changed values to update the base dictionary.

    Returns
    -------
        new_dict : dict
            Dictionary containing values from the merged dictionaries.
    """
    if base_dict is None:
        return update_dict
    base_dict = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if isinstance(value, collections.abc.Mapping):
            base_dict[key] = safe_merge_dicts(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict


##### load configure files
def get_base_config():
    # Load base config
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', "config_common.json"), encoding='utf-8') as f:
        base_config = json.load(f)

    return base_config


def load_config(config=None):
    # Load user config
    print("user config file:", config)
    if isinstance(config, str):
        with open(config, encoding='utf-8') as f:
            user_config = json.load(f)
    elif isinstance(config, dict):
        user_config = config
    else:
        assert config is None, "Config must be None, str, or dict."
        user_config = {}

    # Load task-specific base config
    base_config = get_base_config()

    # Return combined configs
    return safe_merge_dicts(base_config, user_config)

