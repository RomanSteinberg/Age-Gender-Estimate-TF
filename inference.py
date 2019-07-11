import os
import json
from collections import defaultdict
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
from age_gender.utils.dataloader import DataLoader
from age_gender.utils.config_parser import get_config
from age_gender.nets.inception_resnet_v1 import InceptionResnetV1
from age_gender.nets.resnet_v2_50 import ResNetV2_50
from age_gender.utils.model_saver import ModelSaver
from age_gender.utils.dataset_json_loader import DatasetJsonLoader
from age_gender.nets.learning_rate_manager import LearningRateManager


class ModelManager:
    def __init__(self, config):
        self._config = config
        self.batch_size = config['batch_size']
        self.images = tf.placeholder(
            tf.float32, shape=[None, 256, 256, 3])
        self.age_labels = tf.placeholder(tf.int32)
        self.gender_labels = tf.placeholder(tf.int32)
        self.data_init_op = None
        self.init_op = None
        self.model = InceptionResnetV1(phase_train=False, is_training=False)
        self.results = defaultdict(list)
        self.results_names = [
            'file_name',
            'gt_age',
            'gt_gender',
            'pred_age',
            'pred_gender'
        ]

    def inference(self):
        self.create_computational_graph()
        next_data_element, self.data_loader_init_op, self.dataset_size = self.init_data_loader(
            self._config['dataset_path'])
        num_batches = self.dataset_size // self.batch_size
        if (self.dataset_size) % self.batch_size != 0:
            num_batches += 1
        with tf.Graph().as_default() and tf.Session() as sess:
            sess.run(self.init_op)
            sess.run(self.data_loader_init_op)
            saver = ModelSaver()
            saver.restore_model(sess, self._config['model_path'])
            for _ in tqdm(range(num_batches)):
                images, gt_age_labels, gt_gender_labels, file_paths = sess.run(next_data_element)
                self.results['gt_age'] += gt_age_labels.tolist()
                self.results['gt_gender'] += gt_gender_labels.tolist()
                self.results['file_name'] += [s.decode('utf-8') for s in file_paths.tolist()]
                feed_dict = {self.images: images,
                             self.age_labels: gt_age_labels,
                             self.gender_labels: gt_gender_labels,
                             }
                pred_age, pred_gender = sess.run([self.pred_age, self.pred_gender], feed_dict=feed_dict)
                self.results['pred_age'] += pred_age.tolist()
                self.results['pred_gender'] += pred_gender.tolist()
                with open(self._config['results_path'], 'w') as fn:
                    json.dump(self.results, fn)

    def create_computational_graph(self):
        self.variables_to_restore, age_logits, gender_logits = self.model.inference(
            self.images)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        self.pred_age = tf.nn.softmax(age_logits)  # tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        self.pred_gender = tf.nn.softmax(gender_logits)  # tf.argmax(tf.nn.softmax(gender_logits), 1)
        self.init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer())

    def init_data_loader(self, dataset_path):
        dataset_json = json.load(Path(dataset_path).open())
        data_folder = os.path.dirname(dataset_path)
        if Path(self._config['results_path']).is_file():
            prev_result = json.load(open(self._config['results_path'], 'r'))
            for name in self.results_names:
                self.results[name] += prev_result[name]
            processed_files = [fn[fn.find(data_folder)+len(data_folder)+1:] for fn in prev_result['file_name']]
            dataset_json = list(filter(lambda it: it['file_name'] not in processed_files, dataset_json))
        loader = DataLoader(dataset_json, data_folder)
        dataset = loader.create_dataset(
            perform_shuffle=False, batch_size=self.batch_size)
        iterator = tf.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes)
        next_data_element = iterator.get_next()
        data_loader_init_op = iterator.make_initializer(dataset)
        return next_data_element, data_loader_init_op, loader.dataset_len()


if __name__ == '__main__':
    config = get_config('config.yaml', 'inference')
    print(config)
    if not config['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ModelManager(config).inference()
