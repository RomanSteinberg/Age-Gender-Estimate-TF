import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
from age_gender.nets.abstract_net import AbstractNet

class ResNetV2_50(AbstractNet):
    def __init__(self, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.global_step = tf.train.get_or_create_global_step()
        self.bottleneck_scope = 'Head'

    def get_tail(self, images):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            resnet_logits, end_points = resnet_v2.resnet_v2_50(images, is_training=self.is_training)
        return resnet_logits

    def get_age_logits(self, dropout):
        return tf.layers.dense(dropout, self.age_num_classes, activation=None)

    def get_gender_logits(self, dropout):
        return tf.layers.dense(dropout, self.gender_num_classes, activation=None)

    def get_head(self, resnet_logits):
        resnet_flat = tf.layers.flatten(resnet_logits)
        dense2 = tf.layers.dense(resnet_flat, 1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(dense2, 0.75, training=self.is_training)
        return dropout

    def build_model(self, images):
        resnet_logits = self.get_tail(images)
        dropout = self.get_head(resnet_logits)
        return dropout

    def inference(self, images):
        variables_to_restore = [var for var in slim.get_variables_to_restore()]
        with tf.variable_scope('Head'):
            dropout = self.build_model(images)
            age_logits = self.get_age_logits(dropout)
            gender_logits = self.get_gender_logits(dropout)

        return variables_to_restore, age_logits, gender_logits