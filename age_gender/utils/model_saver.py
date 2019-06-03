import os
import yaml
import tensorflow as tf


class ModelSaver(tf.train.Saver):
    def __init__(model_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model_manager = model_manager

    def get_model_checkpoint(self, pretrained_model_folder_or_file):
        model_folder_or_path = pretrained_model_folder_or_file
        if tf.train.checkpoint_exists(model_folder_or_path):
            model_checkpoint = tf.train.latest_checkpoint(model_folder_or_path)
            return model_folder_or_path if not model_checkpoint else model_checkpoint

    def restore_model(self, sess, pretrained_model_folder_or_file):
        if tf.train.checkpoint_exists(pretrained_model_folder_or_file):
            model_checkpoint = self.get_model_checkpoint(pretrained_model_folder_or_file)
            self.restore(sess, model_checkpoint)
            print('Pretrained model loaded')
        else:
            print('No pretrained models found')

    def save_model(self, sess, epoch, experiment_folder):
        save_path = self.save(sess, os.path.join(experiment_folder, "model.ckpt"), global_step=epoch)
        print("Model saved in file: %s" % save_path)