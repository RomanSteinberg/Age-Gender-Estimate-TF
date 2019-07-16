import os
import yaml
import json
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
from datetime import datetime, timedelta
from glob import glob
from age_gender.utils.dataloader import init_data_loader
from age_gender.utils.config_parser import get_config
from age_gender.nets.inception_resnet_v1 import InceptionResnetV1
from age_gender.nets.resnet_v2_50 import ResNetV2_50
from age_gender.utils.model_saver import ModelSaver
from age_gender.nets.learning_rate_manager import LearningRateManager
from age_gender.utils.json_metrics_saver import JsonMetricsWriter

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
        self._learning_rates_config = get_config(config, 'learning_rates')
        learning_rate = self._train_config['learning_rate']
        self.learning_rate_manager = LearningRateManager(
            learning_rate,
            self._learning_rates_config[learning_rate]
        )
        self.model = models[self._train_config['model']](
            **self._models_config[self._train_config['model']])
        self.num_epochs = self._train_config['epochs']
        self.train_size = 0
        self.test_size = None
        self.validation_frequency = None
        self.batch_size = self._train_config['batch_size']
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
        self.images = tf.placeholder(
            tf.float32, shape=[None, 256, 256, 3])
        self.age_labels = tf.placeholder(tf.int32)
        self.gender_labels = tf.placeholder(tf.int32)
        # todo: вынести константы

    def train(self):
        os.makedirs(self.experiment_folder, exist_ok=True)
        log_dir = os.path.join(self.experiment_folder, 'logs')
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
        self.train_metrics_deque = self.train_json_metrics_writer.create_metric_deque()
        self.test_metrics_deque = self.test_json_metrics_writer.create_metric_deque()
        next_data_element, self.train_init_op, self.train_size = init_data_loader(
            self.batch_size,
            self._dataset_config['train_desc_path'],
            self._dataset_config['images_path'],
            self._dataset_config['balance'],
            self.num_epochs
        )
        next_test_data, self.test_init_op, self.test_size = init_data_loader(
            self.batch_size,
            self._dataset_config['test_desc_path'],
            self._dataset_config['images_path'],
            self._dataset_config['balance'],
            self.val_frequency * self.batch_size
        )
        num_batches = self.train_size // self.batch_size + \
            (self.train_size % self.batch_size != 0)
        print(f'Train size: {self.train_size}, test size: {self.test_size}')
        print(
            f'Epochs in train: {self.num_epochs}, batches in epoch: {num_batches}')
        print(f'Validation frequency {self.val_frequency}')
        # print('train_metrics_names:', self.train_metrics_names)

        with tf.Graph().as_default() and tf.Session() as sess:
            tf.random.set_random_seed(100)
            sess.run(self.init_op)
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = ModelSaver(
                var_list=self.variables_to_restore, max_to_keep=100)
            if self._train_config['model_path'] is not None:
                saver.restore_model(sess, self.model_path)
            else:
                print('start training from zero')
            trained_steps = sess.run(self.global_step)
            print('trained_steps', trained_steps)
            trained_epochs = self.calculate_trained_epochs(
                trained_steps, num_batches)
            print('trained_epochs', trained_epochs)

            start_time = {'train': datetime.now()}
            self.save_hyperparameters(start_time)
            fpaths = list()
            if self.mode == 'start':
                sess.run(self.reset_global_step_op)
                trained_steps = 0
                print('global_step turned to zero')
            sess.run(self.train_init_op)
            for tr_batch_idx in range((1+trained_epochs)*num_batches, (1+trained_epochs+self.num_epochs)*num_batches):
                # start_time.update({'train_epoch': datetime.now()})
                train_images, train_age_labels, train_gender_labels, file_paths = sess.run(
                    next_data_element)
                fpaths += [fp.decode('utf-8') for fp in file_paths]
                feed_dict = {self.train_mode: True,
                             # np.zeros([16, 256, 256, 3])
                             self.images: train_images,
                             self.age_labels: train_age_labels,
                             self.gender_labels: train_gender_labels,
                             }
                _, train_metrics_and_errors, step, bottleneck, regularized_vars = sess.run([self.train_op, self.train_metrics_and_errors,
                                                                                            self.global_step, self.bottleneck, self.regularized_vars],
                                                                                           feed_dict=feed_dict)
                #print('step: ', step)
                self.train_metrics_deque, summaries = get_streaming_metrics(self.train_metrics_deque,
                                                                            train_metrics_and_errors, 'train')
                summary_writer.add_summary(summaries, step)
                self.train_json_metrics_writer.dump(
                    int(step), file_paths, self.train_metrics_deque)

                if (step - trained_steps) % self.val_frequency == 0:
                    start_time.update({'test_epoch': datetime.now()})
                    sess.run([self.test_init_op])
                    for ts_batch_idx in range(1, self.val_frequency+1):
                        test_images, test_age_labels, test_gender_labels, test_file_paths = sess.run(
                            next_test_data)
                        feed_dict = {
                            self.train_mode: False,
                            self.images: test_images,
                            self.age_labels: test_age_labels,
                            self.gender_labels: test_gender_labels
                        }
                        # summary = sess.run(self.test_summary, feed_dict=feed_dict)
                        # train_writer.add_summary(summary, step - num_batches + batch_idx)
                        test_metrics_and_errors = sess.run(
                            self.test_metrics_and_errors, feed_dict=feed_dict)
                        self.test_metrics_deque, summaries = get_streaming_metrics(self.test_metrics_deque,
                                                                                   test_metrics_and_errors, 'test')
                        current_batch_num = int(
                            step) - self.val_frequency + ts_batch_idx
                        summary_writer.add_summary(
                            summaries, current_batch_num)
                        self.test_json_metrics_writer.dump(
                            current_batch_num, test_file_paths, self.test_metrics_deque)
                    t = time_spent(start_time['test_epoch'])
                    print(f'Test takes {t}')
                    t = time_spent(start_time['train'])
                    print(
                        f'Train {tr_batch_idx} batches plus test time take {t}')
                    save_path = saver.save(sess, os.path.join(
                        self.experiment_folder, "model.ckpt"), global_step=tr_batch_idx)
                    # self.save_hyperparameters(start_time)
                    print("Model saved in file: %s" % save_path)

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

    def calculate_trained_epochs(self, trained_steps,  num_batches):
        # return (trained_steps) // num_batches
        return (trained_steps - self.model.trained_steps) // num_batches

    def save_hyperparameters(self, start_time):
        self._train_config['duration'] = time_spent(start_time['train'])
        self._train_config['date'] = datetime.now().strftime("%Y_%m_%d_%H_%M")
        num_hyperparams = len(glob(self.experiment_folder + '/*.yaml'))
        hyperparams_name = "hyperparams.yaml" if num_hyperparams == 0 else f"hyperparams_{num_hyperparams}.yaml"
        json_parameters_path = os.path.join(
            self.experiment_folder, hyperparams_name)
        config = dict()
        config['model'] = self._models_config[self._train_config['model']]
        config['learning_rate'] = self._learning_rates_config[self._train_config['learning_rate']]
        config['dataset'] = self._dataset_config
        config['train'] = self._train_config
        with open(json_parameters_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

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


def get_streaming_metrics(metrics_deque, metrics_and_errors, mode):
    summaries_list = list()
    for name in metrics_deque.keys():
        metric = metrics_and_errors[name]
        if name != 'lr':
            metrics_deque[name].append(metric)
            metric = np.mean(metrics_deque[name])
        summary = summary_pb2.Summary.Value(
            tag=f'{mode}/{name}', simple_value=metric)
        summaries_list.append(summary)
    summaries = summary_pb2.Summary(value=summaries_list)
    return metrics_deque, summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="config.yaml", help="config")
    args = parser.parse_args()
    config = get_config(args.config)
    if not config['train']['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ModelManager(config).train()
