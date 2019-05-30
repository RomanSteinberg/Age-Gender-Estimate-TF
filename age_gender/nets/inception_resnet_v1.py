from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from age_gender.nets.inception_resnet_v1_blocks import block35, reduction_a, block17, reduction_b, block8
from age_gender.nets.abstract_net import AbstractNet

class InceptionResnetV1(AbstractNet):
    def __init__(self, keep_probability=0.8, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None,
                 dropout_keep_prob=0.8, is_training=True):
        super().__init__()
        self.global_step = tf.Variable(0, name='global_step', trainable=True)
        self.bottleneck_scope = 'InceptionResnetV1/Bottleneck'
        self.keep_probability = keep_probability
        self.phase_train = phase_train
        self.bottleneck_layer_size = bottleneck_layer_size
        self.weight_decay = weight_decay
        self.reuse = reuse
        self.dropout_keep_prob = dropout_keep_prob
        self.is_training = is_training

        self.batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            # 'updates_collections': None,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        }

    def get_tail(self, images):
        # 149 x 149 x 32
        net = slim.conv2d(images, 32, 3, stride=2, padding='VALID',
                          scope='Conv2d_1a_3x3')
        # 147 x 147 x 32
        net = slim.conv2d(net, 32, 3, padding='VALID',
                          scope='Conv2d_2a_3x3')
        # 147 x 147 x 64
        net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
        # 73 x 73 x 64
        net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                              scope='MaxPool_3a_3x3')
        # 73 x 73 x 80
        net = slim.conv2d(net, 80, 1, padding='VALID',
                          scope='Conv2d_3b_1x1')
        # 71 x 71 x 192
        net = slim.conv2d(net, 192, 3, padding='VALID',
                          scope='Conv2d_4a_3x3')
        # 35 x 35 x 256
        net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                          scope='Conv2d_4b_3x3')
        # 5 x Inception-resnet-A
        net = slim.repeat(net, 5, block35, scale=0.17)
        # Reduction-A
        with tf.variable_scope('Mixed_6a'):
            net = reduction_a(net, 192, 192, 256, 384)
        # 10 x Inception-Resnet-B
        net = slim.repeat(net, 10, block17, scale=0.10)
        # Reduction-B
        with tf.variable_scope('Mixed_7a'):
            net = reduction_b(net)
        # 5 x Inception-Resnet-C
        net = slim.repeat(net, 5, block8, scale=0.20)
        net = block8(net, activation_fn=None)
        return net

    def get_head(self, net):
        with tf.variable_scope('Logits'):
            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                  scope='AvgPool_1a_8x8')
            net = slim.flatten(net)
            net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training,
                               scope='Dropout')
        net = slim.fully_connected(net, self.bottleneck_layer_size, activation_fn=None,
                                   scope='Bottleneck', reuse=False)
        return net

    def build_model(self, images):
        scope = 'InceptionResnetV1'
        with tf.variable_scope(scope, 'InceptionResnetV1', [images], reuse=self.reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=self.is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):
                    # 149 x 149 x 32
                    net = slim.conv2d(images, 32, 3, stride=2, padding='VALID',
                                      scope='Conv2d_1a_3x3')
                    # 147 x 147 x 32
                    net = slim.conv2d(net, 32, 3, padding='VALID',
                                      scope='Conv2d_2a_3x3')
                    # 147 x 147 x 64
                    net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                    # 73 x 73 x 64
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                          scope='MaxPool_3a_3x3')
                    # 73 x 73 x 80
                    net = slim.conv2d(net, 80, 1, padding='VALID',
                                      scope='Conv2d_3b_1x1')
                    # 71 x 71 x 192
                    net = slim.conv2d(net, 192, 3, padding='VALID',
                                      scope='Conv2d_4a_3x3')
                    # 35 x 35 x 256
                    net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                      scope='Conv2d_4b_3x3')
                    # 5 x Inception-resnet-A
                    net = slim.repeat(net, 5, block35, scale=0.17)
                    # Reduction-A
                    with tf.variable_scope('Mixed_6a'):
                        net = reduction_a(net, 192, 192, 256, 384)
                    # 10 x Inception-Resnet-B
                    net = slim.repeat(net, 10, block17, scale=0.10)
                    # Reduction-B
                    with tf.variable_scope('Mixed_7a'):
                        net = reduction_b(net)
                    # 5 x Inception-Resnet-C
                    net = slim.repeat(net, 5, block8, scale=0.20)
                    net = block8(net, activation_fn=None)
                    net = self.get_head(net)
        return net

    def get_age_logits(self, net):
        return slim.fully_connected(net, 101, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(1e-5),
                                    scope='logits/age', reuse=False)

    def get_gender_logits(self, net):
        return slim.fully_connected(net, 2, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(1e-5),
                                    scope='logits/gender', reuse=False)

    def inference(self, images):
        variables_to_restore = [var for var in slim.get_variables_to_restore()]  # slim.get_variables_to_restore()
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=self.batch_norm_params):
            net = self.build_model(images)
        age_logits = self.get_age_logits(net)
        gender_logits = self.get_gender_logits(net)
        return variables_to_restore, age_logits, gender_logits

