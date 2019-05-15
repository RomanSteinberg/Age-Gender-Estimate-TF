# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# use multi CPU
import multiprocessing
import os
import yaml
import time
import cv2
import dlib
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from imutils.face_utils import FaceAligner
from sklearn.model_selection import train_test_split
from utils.config_parser import get_config
FLAGS = None


def get_area(rect):
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    return (bottom - top) * (right - left)


def normalize_rect(rect):
    infinity = 10 ** 10
    h, w = (infinity, infinity)
    left = rect[0] if rect[0] >= 0 else 0
    top = rect[1] if rect[1] >= 0 else 0
    right = rect[2] if rect[2] < w else w - 1
    bottom = rect[3] if rect[3] < h else h - 1
    return [left, top, right, bottom]


def scale(x_scale, y_scale, rect):
    left, top, right, bottom = rect
    weight = (right - left)/2
    height = (bottom - top)/2
    center_x = (left + right)/2
    center_y = (top + bottom)/2
    left = int(center_x - weight*x_scale)
    top = int(center_y - height*y_scale)
    right = int(center_x + weight*x_scale)
    bottom = int(center_y + height*y_scale)
    return [left, top, right, bottom]


def align_face(config, rect, face_aligner, image, gray_image):
    height_scale = config['height_scale']
    width_scale = config['width_scale']
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    scaled_rect = scale(width_scale, height_scale, [left, top, right, bottom])
    normilized_rect = normalize_rect(scaled_rect)
    dlib_rect = dlib.rectangle(*normilized_rect)
    image_raw = face_aligner.align(image, gray_image, dlib_rect)
    return image_raw


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name, i, config, tfrecords_folder):
    config_general = config['general']
    dataset_path = config_general['dataset_path']
    dataset_folder = os.path.dirname(dataset_path)
    tfrecords_path = config_general['tfrecords_path']

    config_image = config['image']
    image_size = config_image['size']
    face_score_threshold = config_image['face_score_threshold']
    face_area_threshold = config_image['face_area_threshold']

    """Converts a dataset to tfrecords."""
    file_name = data_set.file_name
    genders = data_set.gender
    ages = data_set.age
    face_score = data_set.score
    num_examples = data_set.shape[0]
    print(num_examples)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    shape_predictor = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    face_aligner = FaceAligner(predictor, desiredFaceWidth=image_size)

    error = 0
    total = 0
    full_tfrecords_path = os.path.join(tfrecords_path, tfrecords_folder, name)
    if not os.path.exists(full_tfrecords_path):
        os.makedirs(full_tfrecords_path)
    filename = os.path.join(full_tfrecords_path, name + '-%.3d' % i + '.tfrecords')
    small_images = 0
    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            print(f'{index} from {num_examples}')
            if face_score[index] < face_score_threshold:
                continue
            if ~(0 <= ages[index] <= 100):
                continue
            if np.isnan(genders[index]):
                continue
            try:
                image = cv2.imread(os.path.join(dataset_folder, str(file_name[index][0])), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = detector(image, 2)
                if len(rects) != 1:
                    continue
                else:
                    area = get_area(rects[0])
                    if area < face_area_threshold:
                        small_images+=1
                        continue
                    aligned_face = align_face(config['image'], rects[0], face_aligner, image, gray_image)
                    image_raw = aligned_face.tostring()
            except IOError:  # some files seem not exist in face_data dir
                error = error + 1
                pass
            example = tf.train.Example(features=tf.train.Features(feature={
                'age': _int64_feature(int(ages[index])),
                'gender': _int64_feature(int(genders[index])),
                'image_raw': _bytes_feature(image_raw),
                'file_name': _bytes_feature(str(file_name[index][0]).encode('utf-8'))
                }))
            writer.write(example.SerializeToString())
            total = total + 1

    print("There are ", error, " missing pictures")
    print("Found", total, "valid faces")
    print('Images with too small face: ',small_images)


def main(config):
    config_general = config['general']
    test_size = config_general['test_size']
    cpu_cores = config_general['nworks']
    dataset_path = config_general['dataset_path']
    tfrecords_path = config_general['tfrecords_path']
    tfrecords_folder = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    start_time = time.time()
    dataset = pd.read_json(dataset_path)
    train_sets, test_sets = train_test_split(dataset, test_size=test_size, random_state=2017)
    train_sets.reset_index(drop=True, inplace=True)
    test_sets.reset_index(drop=True, inplace=True)
    train_nums = train_sets.shape[0]
    test_nums = test_sets.shape[0]
    train_idx = np.linspace(0, train_nums, cpu_cores + 1, dtype=np.int)
    test_idx = np.linspace(0, test_nums, cpu_cores + 1, dtype=np.int)
    # multi cpu
    pool = multiprocessing.Pool(processes=cpu_cores)
    for p in range(cpu_cores):
        print(train_sets[train_idx[p]:train_idx[p + 1] - 1].shape)
        pool.apply_async(convert_to,
                         (train_sets[train_idx[p]:train_idx[p + 1] - 1].copy().reset_index(drop=True), 'train', p,
                          config, tfrecords_folder))
    for p in range(cpu_cores):
        pool.apply_async(convert_to,
                         (test_sets[test_idx[p]:test_idx[p + 1] - 1].copy().reset_index(drop=True), 'test', p,
                          config, tfrecords_folder))
    pool.close()
    pool.join()

    json_parameters_path = os.path.join(tfrecords_path, tfrecords_folder, 'config.yaml')
    with open(json_parameters_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    duration = time.time() - start_time
    print("Running %.3f sec All done!" % duration)


if __name__ == '__main__':
    config = get_config('config.yaml')
    main(config['dataset_to_tfrecords'])
