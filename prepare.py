import argparse
from utils.config_parser import get_config
from age_gender.utils.converter import ConverterManager
from age_gender.utils.dataset import train_test_split_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ['convert_dataset', 'split_dataset']
    parser.add_argument('command', type=str, choices= choices, help='command for execution')
    args = parser.parse_args()
    config = get_config('config.yaml')['prepare']
    if args.command == 'convert_dataset':
        ConverterManager(config).run()
    else:
        train_test_split_dataset(config)
