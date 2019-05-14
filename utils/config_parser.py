import yaml
import os
from copy import deepcopy


def set_absolute_paths(d, working_dir):
    for key in d.keys():
        if isinstance(d[key], dict):
            set_absolute_paths(d[key], working_dir)
        else:
            if 'path' in key and os.path.abspath(d[key]) != d[key]:
                d[key] = os.path.join(working_dir, d[key])


def get_config(source='config-default.yaml', subproject=None):
    if isinstance(source, str):
        with open(source, 'r') as stream:
            config = yaml.load(stream)
    elif isinstance(source, dict):
        config = source
    else:
        config = None
        raise TypeError('Unexpected source to load config')

    working_dir = os.path.abspath(config['general']['working_dir'])
    config['general']['working_dir'] = working_dir
    print(working_dir)
    set_absolute_paths(config, working_dir)
    if subproject is None:
        return config

    new_config = {} if config[subproject] is None else deepcopy(config[subproject])
    new_config.update(config['general'])
    return new_config
