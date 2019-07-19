import unittest
import random
import string
from age_gender.utils.dataset_json_loader import DatasetJsonLoader


def file_name_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def generate_dataset_json(ages, size = 100):
    dataset = list()
    for ind, max_age in enumerate(ages):
        min_age = 1 if ind == 0 else ages[ind - 1]
        class_size = size if max_age == 40 else size * random.uniform(0.1,0.9)
        for i in range(int(class_size)):
            dataset.append({'file_name': file_name_generator(), 'age': random.randint(min_age, max_age)})
    return dataset


class DatasetJsonLoaderTestCase(unittest.TestCase):
    def setUp(self):
        ages = [20, 30, 40, 60, 100]
        self.init_dataset = generate_dataset_json(ages)
        weights = DatasetJsonLoader.get_weights(self.init_dataset, ages, (30, 40))
        self.config = {'ages': ages, 'weights': weights}
        self.dataset_json_loader = DatasetJsonLoader(self.config, self.init_dataset)
        self.balanced_dataset = self.dataset_json_loader.get_dataset()

    def test_unique_files(self):
        files = set([item['file_name'] for item in self.init_dataset])
        balanced_files = set([item['file_name'] for item in self.balanced_dataset])
        self.assertTrue(files == balanced_files)

    def test_balance(self):
        weigts = DatasetJsonLoader.get_weights(self.balanced_dataset, self.config['ages'], (30,40))
        for weigt in  weigts:
            self.assertTrue(abs(1 - weigt)< 0.1)
