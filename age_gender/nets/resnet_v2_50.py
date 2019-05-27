import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2


def build_resnet(images_tn, age_num_classes=101, gender_num_classes=2, is_training=True):
    checkpoint_file = '/home/roman/dev/age-gender/models/pretrained_models/resnet50/resnet_v2_50.ckpt'
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        resnet_logits, end_points = resnet_v2.resnet_v2_50(images_tn, is_training=is_training)

    resnet_init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, [var for var in tf.global_variables()])

    with tf.variable_scope('Head'):
        resnet_flat = tf.layers.flatten(resnet_logits)
        dense2 = tf.layers.dense(resnet_flat, 1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(dense2, 0.75, training=is_training)
        age_logits = tf.layers.dense(dropout, age_num_classes, activation=None)
        gender_logits = tf.layers.dense(dropout, gender_num_classes, activation=None)

    # init variables
    init_vars = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Head'))

    def init_fn(sess):
        sess.run(init_vars)
        resnet_init_fn(sess)

    return init_fn, age_logits, gender_logits
