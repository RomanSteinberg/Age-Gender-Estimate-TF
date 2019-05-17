import argparse
from functools import partial

from age_gender.utils.config_parser import get_config
from age_gender.utils.converter import ConverterManager
from age_gender.utils.splitter import train_test_split_dataset

if __name__ == "__main__":
    config = get_config('config.yaml', 'prepare')

    parser = argparse.ArgumentParser()
    choices = {'convert_dataset': ConverterManager(config).run,
               'split_dataset': partial(train_test_split_dataset, config)}
    parser.add_argument('command', type=str, choices=choices.keys(), help='dataset preparation command')
    args = parser.parse_args()

    choices[args.command]()
