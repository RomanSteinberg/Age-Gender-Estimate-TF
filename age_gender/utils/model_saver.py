import os
import yaml
import tensorflow as tf


class ModelSaver(tf.train.Saver):
    def __init__(self, model_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_manager = model_manager

    def get_model_checkpoint(self):
        model_folder_or_path = self.model_manager.pretrained_model_folder_or_file
        if tf.train.checkpoint_exists(model_folder_or_path):
            model_checkpoint = tf.train.latest_checkpoint(model_folder_or_path)
            return model_folder_or_path if not model_checkpoint else model_checkpoint

    def restore_model(self, sess):
        if tf.train.checkpoint_exists(self.model_manager.pretrained_model_folder_or_file):
            model_checkpoint = self.get_model_checkpoint()
            self.restore(sess, model_checkpoint)
            print('Pretrained model loaded')
        else:
            print('No pretrained models found')

    def save_model(self, sess, epoch):
        save_path = self.save(sess, os.path.join(self.model_manager.experiment_folder, "model.ckpt"), global_step=epoch)
        self.save_hyperparameters()
        print("Model saved in file: %s" % save_path)

    def save_hyperparameters(self):
        self.model_manager.config['duration'] = self.model_manager.time_spent(self.model_manager.start_time['train'])
        json_parameters_path = os.path.join(self.model_manager.experiment_folder, "hyperparams.yaml")
        with open(json_parameters_path, 'w') as file:
            yaml.dump(self.model_manager.config, file, default_flow_style=False)
