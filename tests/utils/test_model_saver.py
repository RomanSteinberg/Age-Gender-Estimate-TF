import unittest
import os
import numpy as np
import tensorflow as tf
from age_gender.utils.config_parser import get_config
from age_gender.utils.model_saver import ModelSaver
from age_gender.nets.inception_resnet_v1 import InceptionResnetV1
from age_gender.utils.converter import Converter


dataset = [{'age': 25, 'file_name': '57/nm2799457_rm1351197440_1987-9-23_2013.jpg', 'gender': 1},
           {'age': 19, 'file_name': '98/nm3794498_rm3574828800_1983-1-1_2002.jpg', 'gender': 1},
           {'age': 32, 'file_name': '32/nm0890232_rm1198966784_1968-11-27_2001.jpg', 'gender': 1},
           {'age': 32, 'file_name': '32/nm0890232_rm2843133952_1968-11-27_2001.jpg', 'gender': 1},
           {'age': 27, 'file_name': '09/nm0890809_rm2228126976_1979-11-18_2007.jpg', 'gender': 0}]


def load_network(pretrained_model_folder_or_file):
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='input_image')
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
    model = InceptionResnetV1(phase_train=False, is_training=False)
    variables_to_restore, age_logits, gender_logits = model.inference(images_norm)
    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = ModelSaver(variables_to_restore)
    saver.restore_model(sess, pretrained_model_folder_or_file)
    return sess, age, gender, images_pl


class ModelSaverTestCase(unittest.TestCase):
    def setUp(self):
        self.config = get_config('config.yaml', 'validate')
        converter_config = get_config('config.yaml', 'prepare')
        self.converter = Converter(converter_config)
        if not self.config['cuda']:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        self.sess, self.age, self.gender, self.images_pl = load_network(self.config['pretrained_model_folder_or_file'])

    def test_restoring_inception_resnet(self):
        path_to_images = os.path.dirname(self.config['dataset_path'])
        faces = np.empty((len(dataset), 256, 256, 3))
        for ind, record in enumerate(dataset):
            image_path = os.path.join(path_to_images, record['file_name'])
            face = self.converter.convert_image(image_path)
            faces[ind, :, :, :] = face
        _ , genders = self.sess.run([self.age, self.gender], feed_dict={self.images_pl: faces})
        expected_genders = [1, 0, 1, 0, 1]
        self.assertListEqual(genders.tolist(), expected_genders)

