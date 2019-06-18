import os
import yaml
import json
from pathlib import Path
import tensorflow as tf
from datetime import datetime, timedelta
from age_gender.utils.dataloader import DataLoader
from age_gender.utils.config_parser import get_config
from age_gender.nets.inception_resnet_v1 import InceptionResnetV1
from age_gender.nets.resnet_v2_50 import ResNetV2_50
from age_gender.utils.model_saver import ModelSaver
from age_gender.utils.dataset_json_loader import DatasetJsonLoader
from age_gender.nets.learning_rate_manager import LearningRateManager

models = {'inception_resnet_v1' : InceptionResnetV1, 'resnet_v2_50': ResNetV2_50}

class ModelManager:
    def __init__(self, config):
        # parameters
        self._config = config
        self.learning_rate_manager = LearningRateManager(self._config['init']['learning_rate'])
        self.model = models[config['init']['model']]()
        self.pretrained_model_folder_or_file = config['init']['pretrained_model_folder_or_file']
        self.num_epochs = config['epochs']
        self.train_size = 0
        self.test_size = 0
        self.batch_size = config['batch_size']
        self.save_frequency = config['init']['save_frequency']

        self.mode = config['init']['mode']
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
        self.images = tf.placeholder(tf.float32, shape=[self.batch_size, 256, 256, 3])
        self.age_labels = tf.placeholder(tf.int32)
        self.gender_labels = tf.placeholder(tf.int32)
        # todo: вынести константы

    def train(self):
        os.makedirs(self.experiment_folder, exist_ok=True)
        log_dir = os.path.join(self.experiment_folder, 'logs')
        self.create_computational_graph()
        next_data_element, self.train_init_op, self.train_size = self.init_data_loader('train')
        next_test_data, self.test_init_op, self.test_size = self.init_data_loader('test')

        num_batches = (self.train_size + 1) // self.batch_size
        print(f'Train size: {self.train_size}, test size: {self.test_size}')
        print(f'Epochs in train: {self.num_epochs}, batches in epoch: {num_batches}')

        with tf.Graph().as_default() and tf.Session() as sess:
            tf.random.set_random_seed(100)
            sess.run(self.init_op)
            train_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = ModelSaver(var_list=self.variables_to_restore, max_to_keep=100)
            saver.restore_model(sess, self.pretrained_model_folder_or_file)
            trained_steps = sess.run(self.global_step)
            print('trained_steps', trained_steps)
            trained_epochs = self.calculate_trained_epochs(trained_steps, num_batches)
            print('trained_epochs', trained_epochs)
            #exit(0)
            start_time = {'train': datetime.now()}
            fpaths = list()
            for epoch in range(1+trained_epochs, 1 + trained_epochs + self.num_epochs):
                sess.run(self.train_init_op)
                start_time.update({'train_epoch': datetime.now()})
                for batch_idx in range(num_batches):
                    train_images, train_age_labels, train_gender_labels, file_paths = sess.run(next_data_element)
                    fpaths += [fp.decode('utf-8') for fp in file_paths]
                    feed_dict = {self.train_mode: True,
                                 self.images: train_images,  # np.zeros([16, 256, 256, 3])
                                 self.age_labels: train_age_labels,
                                 self.gender_labels: train_gender_labels}
                    _, summary, step = sess.run([self.train_op, self.train_summary, self.global_step],
                                                feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)

                t = time_spent(start_time['train_epoch'])
                print(f'Train epoch {epoch} takes {t}')

                if epoch % self.save_frequency == 0 or epoch == 1:
                    start_time.update({'test_epoch': datetime.now()})
                    sess.run([self.test_init_op])
                    for batch_idx in range((self.test_size + 1) // self.batch_size):
                        test_images, test_age_labels, test_gender_labels, _ = sess.run(next_test_data)
                        feed_dict = {self.train_mode: False,
                                     self.images: test_images,
                                     self.age_labels: test_age_labels,
                                     self.gender_labels: test_gender_labels}
                        summary = sess.run(self.test_summary, feed_dict=feed_dict)
                        train_writer.add_summary(summary, step - num_batches + batch_idx)
                    t = time_spent(start_time['test_epoch'])
                    print(f'Test epoch {epoch} takes {t}')
                    save_path = saver.save(sess, os.path.join(self.experiment_folder, "model.ckpt"), global_step=epoch)
                    self.save_hyperparameters(start_time)
                    print("Model saved in file: %s" % save_path)

            saver.save_model(sess, epoch, self.experiment_folder)
            self.save_hyperparameters(start_time)

    def get_experiment_folder(self, mode):
        if mode == 'start':
            working_dir = config['working_dir']
            experiment_folder = os.path.join(working_dir, 'experiments', datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))
            os.makedirs(experiment_folder, exist_ok=True)
        elif mode == 'continue':
            experiment_folder = \
                self.pretrained_model_folder_or_file if os.path.isdir(self.pretrained_model_folder_or_file) else \
                    os.path.dirname(self.pretrained_model_folder_or_file)
        else:
            experiment_folder = 'experiments'
        return experiment_folder

    def calculate_trained_epochs(self, trained_steps,  num_batches):
        # return (trained_steps) // num_batches
        return (trained_steps - self.model.trained_steps) // num_batches

    def save_hyperparameters(self, start_time):
        self._config['duration'] = time_spent(start_time['train'])
        json_parameters_path = os.path.join(self.experiment_folder, "hyperparams.yaml")
        with open(json_parameters_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

    def create_computational_graph(self):
        self.variables_to_restore, age_logits, gender_logits = self.model.inference(self.images)
        # head
        age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.age_labels, logits=age_logits)
        age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)
        gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.gender_labels,
                                                                              logits=gender_logits)
        gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

        # l2 regularization
        total_loss = tf.add_n(
            [gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        abs_loss = tf.losses.absolute_difference(self.age_labels, age)
        gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, self.gender_labels, 1), tf.float32))

        self.reset_global_step_op = tf.assign(self.global_step, 0)

        lr = self.learning_rate_manager.get_learning_rate(self.global_step)

        metrics_and_errors = [abs_loss, age_cross_entropy_mean, gender_acc, gender_cross_entropy_mean, total_loss]
        self.train_summary = self.define_summaries(metrics_and_errors, lr, 'train')
        self.test_summary = self.define_summaries(metrics_and_errors, None, 'test')

        optimizer = tf.train.AdamOptimizer(lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(total_loss, self.global_step)

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def init_data_loader(self, mode):
        dataset_path = self._config['init'][f'{mode}_dataset_path']
        dataset_json_config = self._config['init']['dataset_json_loader']
        dataset_json = json.load(Path(dataset_path).open())
        if self._config['init']['balance_dataset']:
            dataset_json_loader = DatasetJsonLoader(dataset_json_config, dataset_json)
            dataset_json = dataset_json_loader.get_dataset()
        data_folder = os.path.dirname(dataset_path)
        loader = DataLoader(dataset_json, data_folder)
        dataset = loader.create_dataset(perform_shuffle=True, batch_size=self.batch_size)

        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        next_data_element = iterator.get_next()
        training_init_op = iterator.make_initializer(dataset)
        return next_data_element, training_init_op, loader.dataset_len()

    def define_summaries(self, metrics, lr, mode):
        summaries = list()

        def helper(name, tn):
            summaries.append(tf.summary.scalar(f'{mode}/{name}', tn))

        names = ['train_abs_age_error', 'age_cross_entropy', 'gender_accuracy', 'gender_cross_entropy', 'total_loss']
        for name, tn in zip(names, metrics):
            helper(name, tn)

        if mode == 'train':
            helper('lr', lr)
            bottleneck = [v for v in
                          tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model.bottleneck_scope)]
            for var in bottleneck:
                summaries.append(tf.summary.histogram(var.name, var))
        return tf.summary.merge(summaries)

def time_spent(start):
    sec = int((datetime.now() - start).total_seconds())
    return str(timedelta(seconds=sec))


if __name__ == '__main__':
    config = get_config('config.yaml', 'train')
    if not config['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ModelManager(config).train()
