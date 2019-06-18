import os
import cv2
import dlib
import json
import numpy as np
import tensorflow as tf
from age_gender.utils.config_parser import get_config
from age_gender.utils.model_saver import ModelSaver
from age_gender.nets.inception_resnet_v1 import InceptionResnetV1
# from age_gender.nets.old_inception import inference
# from age_gender.nets.new_inception import inference as new_inference, Inception
from age_gender.utils.converter import Converter
from age_gender.utils.dataloader import DataLoader
import argparse

def validate(config, converter, sess, age, gender, images_pl):
    dataset_path = config['dataset_path']
    with open(dataset_path) as f:
        dataset_json = json.load(f)
    dataset = dataset_json[:5]
    print(dataset)

    # for batch_idx in range((test_size + 1) // self.batch_size):
    #     test_images, test_age_labels, test_gender_labels, _ = sess.run(next_test_data)
    path_to_images = os.path.dirname(dataset_path)
    faces = np.empty((len(dataset), 256, 256, 3))
    for ind, record in enumerate(dataset):
        image_path = os.path.join(path_to_images, record['file_name'])
        face = converter.convert_image(image_path)
        faces[ind, :, :, :] = face

    ages, genders = sess.run([age, gender], feed_dict={images_pl: faces})
    print('ages: ', ages)
    print('genders: ', genders)

def load_network_new(config):
    pretrained_model_folder_or_file = config['pretrained_model_folder_or_file']
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 256, 256 ,3], name='input_image')
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

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, help='path to json for dataset')
    # parser.add_argument('--images', type=str, help='path to images')
    # parser.add_argument('--output', type=str, help='path to output json')
    # args = parser.parse_args()

    config = get_config('config.yaml','validate')
    converter_config = get_config('config.yaml','prepare')
    converter = Converter(converter_config)
    if not config['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess, age, gender, images_pl = load_network_new(config)
    validate(config, converter, sess,  age, gender, images_pl)

def get_results(predictions, type, size):
