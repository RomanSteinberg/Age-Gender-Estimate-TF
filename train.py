import os
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from glob import glob
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from age_gender.utils.dataloader import init_data_loader
from age_gender.utils.config_parser import get_config
from age_gender.nets.inception_resnet_v1 import InceptionResnetV1
from age_gender.nets.resnet_v2_50 import ResNetV2_50
from age_gender.utils.model_saver import ModelSaver
from age_gender.nets.learning_rate_manager import LearningRateManager
from age_gender.utils.json_metrics_saver import JsonMetricsWriter
from age_gender.utils.logger import Logger

models = {'inception_resnet_v1': InceptionResnetV1,
          'resnet_v2_50': ResNetV2_50}


class ModelManager:
    def __init__(self, config):
        # parameters
        self._models_config = get_config(config, 'models')
        self._train_config = get_config(config, 'train')
        self._dataset_config = get_config(config, 'datasets')[
            self._train_config['dataset']]
        if not self._train_config['balance_dataset']:
            self._dataset_config['balance'] = None
        lr_method = self._train_config['learning_rate']
        lr_config = get_config(config, 'learning_rates')
        self._lr_config = lr_config['linear'] if lr_method == 'test_lr' else lr_config[lr_method]
        self.learning_rate_manager = LearningRateManager(lr_method,self._lr_config)
        self.model = models[self._train_config['model']](
            **self._models_config[self._train_config['model']])
        self.num_epochs = self._train_config['epochs']
        self.train_size = 0
        self.test_size = None
        self.batch_size = self._train_config['batch_size']
        self.save_frequency = self._train_config['save_frequency']
        self.val_frequency = self._train_config['val_frequency']
        self.mode = self._train_config['mode']
        self.model_path = self._train_config['model_path']
        self.experiment_folder = self.get_experiment_folder(self.mode)

        # operations
        self.global_step = self.model.global_step
        self.train_mode = tf.placeholder(tf.bool)
        self.init_op = None
        self.train_op = None
        self.reset_global_step_op = None
        self.train_summary = None
        self.train_init_op = None
        self.test_summary = None
        self.test_init_op = None
        self.images = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        self.age_labels = tf.placeholder(tf.int32)
        self.gender_labels = tf.placeholder(tf.int32)
        self.test_lr = list() if self.learning_rate_manager.method_name == 'test_lr' else None
        # todo: вынести константы

    def train(self):
        os.makedirs(self.experiment_folder, exist_ok=True)
        logs_folder = os.path.join(self.experiment_folder, 'logs')
        logger = Logger('train', logs_folder)
        logger.info('Train starts')
        self.create_computational_graph()
        self.train_json_metrics_writer = JsonMetricsWriter(
            str(Path(self.experiment_folder).joinpath('train_metrics.json')),
            self.train_metrics_and_errors.keys(),
            self.val_frequency
        )
        self.test_json_metrics_writer = JsonMetricsWriter(
            str(Path(self.experiment_folder).joinpath('test_metrics.json')),
            self.test_metrics_and_errors.keys(),
            self.val_frequency
        )
        train_metrics_and_errors = dict()
        self.train_metrics_deque = self.train_json_metrics_writer.create_metric_deque()
        self.test_metrics_deque = self.test_json_metrics_writer.create_metric_deque()
        next_data_element, self.train_init_op, self.train_size = init_data_loader(
            batch_size=self.batch_size,
            desc_path=self._dataset_config['train_desc_path'],
            images_path=self._dataset_config['images_path'],
            balance_config=self._dataset_config['balance'],
            epochs=self.num_epochs,
            num_prefetch=self._train_config['num_prefetch'],
            num_parallel_calls=self._train_config['num_parallel_calls']
        )
        next_test_data, self.test_init_op, self.test_size = init_data_loader(
            batch_size=self.batch_size,
            desc_path=self._dataset_config['test_desc_path'],
            images_path=self._dataset_config['images_path'],
            balance_config=self._dataset_config['balance'],
            min_size=self.num_epochs*self.train_size,
            num_prefetch=self._train_config['num_prefetch'],
            num_parallel_calls=self._train_config['num_parallel_calls']
        )
        num_batches = self.train_size // self.batch_size + (self.train_size % self.batch_size != 0)
        print(f'Train size: {self.train_size}, test size: {self.test_size}')
        print(f'Epochs in train: {self.num_epochs}, batches in epoch: {num_batches}')
        print(f'Validation frequency {self.val_frequency}')
        # print('train_metrics_names:', self.train_metrics_names)

        with tf.Graph().as_default() and tf.Session() as sess:
            tf.random.set_random_seed(100)
            sess.run(self.init_op)
            summary_writer = tf.summary.FileWriter(logs_folder, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = ModelSaver(var_list=self.variables_to_restore, max_to_keep=100)
            if self._train_config['model_path'] is not None:
                self._train_config['model_path'] = saver.restore_model(sess, self.model_path)
            else:
                print('start training from zero')
            # todo: trained steps should consider loading Boyan and loading our own model
            trained_steps = sess.run(self.global_step)
            print('trained_steps', trained_steps)
            # fpaths = list()
            if self.mode == 'start':
                sess.run(self.reset_global_step_op)
                trained_steps = 0
                print('global_step turned to zero')

            sess.run(self.train_init_op)
            sess.run(self.test_init_op)
            start_time = {'train': datetime.now()}
            logger.info('Initialization complete. Train loop starts.')
            self.save_hyperparameters(start_time)
            for tr_batch_idx in range(1+trained_steps, 1+trained_steps+self.num_epochs*num_batches):
                if train_metrics_and_errors.get('lr', 1.) <= 10 ** -10:
                    print(f'Learning rate == 0 at {tr_batch_idx} step')
                    break
                # start_time.update({'train_epoch': datetime.now()})
                logger.debug('load train batch')
                train_images, train_age_labels, train_gender_labels, file_paths = sess.run(next_data_element)
                logger.debug('train on batch')
                # fpaths += [fp.decode('utf-8') for fp in file_paths]

                operations = [self.train_op, self.train_metrics_and_errors, self.global_step,
                              self.bottleneck, self.regularized_vars]
                feed_dict = {
                    self.train_mode: True,
                    self.images: train_images,
                    self.age_labels: train_age_labels,
                    self.gender_labels: train_gender_labels,
                }
                _, train_metrics_and_errors, step, bottleneck, regularized_vars = sess.run(operations,
                                                                                           feed_dict=feed_dict)
                logger.debug('calc train streaming metrics')
                summaries = get_streaming_metrics(self.train_metrics_deque, train_metrics_and_errors, 'train')
                logger.debug('save train summaries')
                summary_writer.add_summary(summaries, step)
                # self.train_json_metrics_writer.dump(int(step), file_paths, self.train_metrics_deque)
                logger.debug('train iteration complete')

                if (step - trained_steps) % self.save_frequency == 0:
                    save_path = saver.save(sess, os.path.join(
                        self.experiment_folder, 'model.ckpt'), global_step=tr_batch_idx)
                    print(f'Model saved in file: {save_path}')
                
                if (step - trained_steps) % self.val_frequency == 0:
                    last_lr = train_metrics_and_errors['lr']
                    start_time.update({'test_epoch': datetime.now()})
                    for ts_batch_idx in range(1, self.val_frequency+1):
                        logger.debug('load test batch')
                        test_images, test_age_labels, test_gender_labels, test_file_paths = sess.run(next_test_data)
                        logger.debug('test on batch')
                        feed_dict = {
                            self.train_mode: False,
                            self.images: test_images,
                            self.age_labels: test_age_labels,
                            self.gender_labels: test_gender_labels
                        }

                        test_metrics_and_errors = sess.run(self.test_metrics_and_errors, feed_dict=feed_dict)
                        test_metrics_and_errors['lr'] = last_lr
                        logger.debug('calc test streaming metrics')
                        summaries = get_streaming_metrics(self.test_metrics_deque, test_metrics_and_errors, 'test', self.test_lr)
                        logger.debug('save test summaries')
                        current_batch_num = int(step) - self.val_frequency + ts_batch_idx
                        summary_writer.add_summary(summaries, current_batch_num)
                        # self.test_json_metrics_writer.dump(current_batch_num, test_file_paths, self.test_metrics_deque)
                        logger.debug('test iteration complete')

                    self.save_test_lr_data()

                    t = time_spent(start_time['test_epoch'])
                    print(f'Test takes {t}')
                    t = time_spent(start_time['train'])
                    print(
                        f'Train {tr_batch_idx} batches plus test time take {t}')

            saver.save_model(sess, tr_batch_idx, self.experiment_folder)

    def get_experiment_folder(self, mode):
        if mode == 'start':
            working_dir = self._train_config['working_dir']
            experiment_folder = os.path.join(
                working_dir, 'experiments', datetime.now().strftime("%Y_%m_%d_%H_%M"))
            os.makedirs(experiment_folder, exist_ok=True)
        elif mode == 'continue':
            experiment_folder = \
                self.model_path if os.path.isdir(self.model_path) else \
                os.path.dirname(self.model_path)
        else:
            experiment_folder = 'experiments'
        return experiment_folder

    def save_hyperparameters(self, start_time):
        self._train_config['duration'] = time_spent(start_time['train'])
        self._train_config['date'] = datetime.now().strftime("%Y_%m_%d_%H_%M")
        config = dict()
        config['model'] = self._models_config[self._train_config['model']]
        config['learning_rate'] = self._lr_config
        config['dataset'] = self._dataset_config
        config['train'] = self._train_config

        num_hyperparams = len(glob(self.experiment_folder + '/*.yaml'))
        hyperparams_name = "hyperparams.yaml" if num_hyperparams == 0 else f"hyperparams_{num_hyperparams}.yaml"
        fn = os.path.join(self.experiment_folder, hyperparams_name)
        with open(fn, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

    def save_test_lr_data(self):
        if self.learning_rate_manager.method_name == 'test_lr':
            fn = os.path.join(self.experiment_folder, 'test_lr.json')
            with open(fn, 'w') as file:
                json.dump(self.test_lr, file)

    def create_computational_graph(self):
        self.variables_to_restore, age_logits, gender_logits = self.model.inference(
            self.images)
        # head
        age_label_encoded = tf.one_hot(indices=self.age_labels, depth=101)
        age_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=age_label_encoded,
                                                                    logits=age_logits)
        age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)
        gender_labels_encoded = tf.one_hot(indices=self.gender_labels, depth=2)
        gender_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=gender_labels_encoded,
                                                                       logits=gender_logits)
        gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

        # l2 regularization
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(
            tf.nn.softmax(age_logits), age_), axis=1)
        mae = tf.losses.absolute_difference(self.age_labels, age)
        mse = tf.losses.mean_squared_error(self.age_labels, age)
        gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(
            gender_logits, self.gender_labels, 1), tf.float32))

        total_loss = tf.add_n(
            [gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.regularized_vars = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reset_global_step_op = tf.assign(self.global_step, 0)

        lr = self.learning_rate_manager.get_learning_rate(self.global_step)

        metrics_and_errors = {
            'mae': mae,
            'mse': mse,
            'age_cross_entropy_mean': age_cross_entropy_mean,
            'gender_acc': gender_acc,
            'gender_cross_entropy_mean': gender_cross_entropy_mean,
            'total_loss': total_loss
        }
        self.metrics_and_errors = metrics_and_errors
        self.test_metrics_and_errors = self.metrics_and_errors
        self.train_metrics_and_errors = self.metrics_and_errors.copy()
        self.train_metrics_and_errors.update({'lr': lr})

        optimizer = tf.train.AdamOptimizer(lr)
        # update batch normalization layer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(total_loss, self.global_step)

        self.init_op = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer())

        self.bottleneck = [v for v in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model.bottleneck_scope)]


def time_spent(start):
    sec = int((datetime.now() - start).total_seconds())
    return str(timedelta(seconds=sec))


def get_streaming_metrics(metrics_deque, metrics_and_errors, mode, test_lr=None):
    summaries_list = list()
    test_lr_chunk = dict()
    for name in metrics_deque.keys():
        metric = metrics_and_errors[name]
        if name != 'lr':
            metrics_deque[name].append(metric)
            metric = np.mean(metrics_deque[name])
        test_lr_chunk[name] = float(metric)
        summary = summary_pb2.Summary.Value(tag=f'{mode}/{name}', simple_value=metric)
        summaries_list.append(summary)
    if test_lr is not None:
        test_lr_chunk['lr'] = float(metrics_and_errors['lr'])  # необходимо значение lr взятое из train стадии
        test_lr.append(test_lr_chunk)
    summaries = summary_pb2.Summary(value=summaries_list)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="config")
    args = parser.parse_args()
    config = get_config(args.config)
    if not config['train']['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if config['train']['mode'] not in ['start', 'continue', 'test']:
        raise ValueError('Invalid mode!')

    ModelManager(config).train()
