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
        self.trained_steps = 5136169

    def get_tail(self):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            net, _ = resnet_v2.resnet_v2_50(self.images, is_training=self.is_training)
        return net

    def get_head(self, net):
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
        net = tf.layers.dropout(net, 0.75, training=self.is_training)
        return net

    def get_age_logits(self, dropout):
        return tf.layers.dense(dropout, self.age_num_classes, activation=None)

    def get_gender_logits(self, dropout):
        return tf.layers.dense(dropout, self.gender_num_classes, activation=None)

    def inference(self, images):
        self.images = images
        variables_to_restore = [var for var in slim.get_variables_to_restore()]

        with tf.variable_scope('Head'):
            net = self.build_model()
            age_logits = self.get_age_logits(net)
            gender_logits = self.get_gender_logits(net)

        return variables_to_restore, age_logits, gender_logits