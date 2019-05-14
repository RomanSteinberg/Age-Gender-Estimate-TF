import argparse
from utils.config_parser import get_config
from age_gender.utils.converter import Converter

if __name__ == "__main__":
    config = get_config('config.yaml')['prepare']
    Converter(config).convert_dataset()
