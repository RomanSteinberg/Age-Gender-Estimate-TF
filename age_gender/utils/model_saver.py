import os
from datetime import datetime

import yaml
import tensorflow as tf


class ModelSaver(tf.train.Saver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def restore_model(self, sess, model_folder_or_path):
        model_checkpoint = self.get_model_checkpoint(model_folder_or_path)
        if model_checkpoint is not None:
            self.restore(sess, model_checkpoint)
            print('Pretrained model loaded')
        else:
            print('No pretrained models found')

    def save_model(self, sess, epoch, experiment_folder):
        save_path = self.save(sess, os.path.join(experiment_folder, "model.ckpt"), global_step=epoch)
        print("Model saved in file: %s" % save_path)

    @staticmethod
    def get_model_checkpoint(model_folder_or_path):
        if tf.train.checkpoint_exists(model_folder_or_path):
            model_checkpoint = tf.train.latest_checkpoint(model_folder_or_path)
            return model_folder_or_path if not model_checkpoint else model_checkpoint
        return None

    @staticmethod
    def save_hyperparameters(experiment_folder, duration, config):
        config['duration'] = duration
        config['date'] = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        json_parameters_path = os.path.join(experiment_folder, "hyperparams.yaml")
        with open(json_parameters_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)