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

import argparse
import math
# use multi CPU
import multiprocessing
import os
import time

import cv2
import dlib
import numpy as np
import pandas as pd
import tensorflow as tf
from imutils.face_utils import FaceAligner
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

FLAGS = None

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
    center = (int(center_x), int(center_y))

    left = int(center_x - weight*x_scale)
    top = int(center_y - height*y_scale)
    right = int(center_x + weight*x_scale)
    bottom = int(center_y + height*y_scale)
    return [left, top, right, bottom]

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name, i, base_path):
    """Converts a dataset to tfrecords."""
    file_name = data_set.file_name
    genders = data_set.gender
    ages = data_set.age
    face_score = data_set.score
    second_face_score = data_set.second_score
    num_examples = data_set.shape[0]
    print(num_examples)
    image_base_dir = os.path.join(base_path, "imdb_crop")

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    shape_predictor = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    error = 0
    total = 0
    tfrecords_path = os.path.join(base_path, name)
    if not os.path.exists(tfrecords_path):
        os.mkdir(tfrecords_path)
    filename = os.path.join(tfrecords_path, name + '-%.3d' % i + '.tfrecords')
    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            print(f'{index} from {num_examples}')
            if face_score[index] < 0.75:
                continue
            # if (~np.isnan(second_face_score[index])) and second_face_score[index] > 0.0:
            #     continue
            if ~(0 <= ages[index] <= 100):
                continue

            if np.isnan(genders[index]):
                continue

            try:
                image = cv2.imread(os.path.join(image_base_dir, str(file_name[index][0])), cv2.IMREAD_COLOR)
                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = detector(image, 2)
                if len(rects) != 1:
                    continue
                else:
                    rect = rects[0]
                    left = rect.left()
                    top = rect.top()
                    right = rect.right()
                    bottom = rect.bottom()
                    scaled_rect = scale(1.3, 1.5, [left,top,right,bottom])
                    normilized_rect = normalize_rect(scaled_rect)
                    dlib_rect = dlib.rectangle(*normilized_rect)
                    image_raw = fa.align(image, image, dlib_rect)
                    image_raw = image_raw.tostring()
            except IOError:  # some files seem not exist in face_data dir
                error = error + 1
                pass
            example = tf.train.Example(features=tf.train.Features(feature={
                'age': _int64_feature(int(ages[index])),
                'gender': _int64_feature(int(genders[index])),
                'image_raw': _bytes_feature(image_raw),
                'file_name': _bytes_feature(str.encode(str(file_name[index][0])))
                }))
            writer.write(example.SerializeToString())
            total = total + 1
    print("There are ", error, " missing pictures")
    print("Found", total, "valid faces")

def main(test_size, cpu_cores, tfrecords_bath_dir):
    start_time = time.time()
    data_sets = pd.read_json('./data/imdb_crop/imdb.json')
    train_sets, test_sets = train_test_split(data_sets, test_size=test_size, random_state=2017)
    train_sets.reset_index(drop=True, inplace=True)
    test_sets.reset_index(drop=True, inplace=True)
    train_nums = train_sets.shape[0]
    test_nums = test_sets.shape[0]
    train_idx = np.linspace(0, train_nums, cpu_cores + 1, dtype=np.int)
    test_idx = np.linspace(0, test_nums, math.ceil(cpu_cores / 4.0) + 1, dtype=np.int)
    # multi cpu
    pool = multiprocessing.Pool(processes=cpu_cores)
    for p in range(cpu_cores):
        print(train_sets[train_idx[p]:train_idx[p + 1] - 1].shape)
        pool.apply_async(convert_to,
                         (train_sets[train_idx[p]:train_idx[p + 1] - 1].copy().reset_index(drop=True), 'train', p,
                          tfrecords_bath_dir,))
    for p in range(cpu_cores):
        pool.apply_async(convert_to,
                         (test_sets[test_idx[p]:test_idx[p + 1] - 1].copy().reset_index(drop=True), 'test', p,
                          tfrecords_bath_dir,))
    pool.close()
    pool.join()
    duration = time.time() - start_time
    print("Running %.3f sec All done!" % duration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data", help="Base path of datasets and tfrecords")
    parser.add_argument("--nworks", default=8, type=int, help="Use n cores to create tfrecords at a time")
    parser.add_argument("--test_size", type=float, default=0.2, help="How many items as testset")
    args = parser.parse_args()
    main(test_size=args.test_size, cpu_cores=args.nworks, tfrecords_bath_dir=args.base_path)