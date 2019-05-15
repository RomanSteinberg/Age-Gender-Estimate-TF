import pandas as pd
import os
from sklearn.model_selection import train_test_split

def train_test_split_dataset(config):
    config = config['general']
    test_size = config['test_size']
    json_path = config['json_path']
    dataset_path = config['processed_dataset_path']
    dataset = pd.read_json(json_path)
    train, test = train_test_split(dataset, test_size=test_size)
    train_json_path = os.path.join(dataset_path,'train.json')
    train.to_json(train_json_path,orient='records')
    test_json_path = os.path.join(dataset_path, 'test.json')
    test.to_json(test_json_path, orient='records')
    if os.path.exists(train_json_path) and os.path.exists(test_json_path):
        print('done')
    else:
        print('something went wrong!')



