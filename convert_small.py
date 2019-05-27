from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from datetime import timedelta
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from time import monotonic as now

from concurrent.futures import ProcessPoolExecutor, as_completed


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class Converter:

    def __init__(self, json_path, dataset_path, tfrecords_path, slice, pid):
        self.dataset_path = dataset_path
        self.tfrecords_path = tfrecords_path
        self.slice = slice
        self.pid = pid
        self.json_path = json_path

    def convert_dataset(self):
        with open(self.json_path) as f:
            data_set_json = json.load(f)[self.slice[0]: self.slice[1]]
        data_set = pd.DataFrame(data_set_json)
        file_name = data_set.file_name
        genders = data_set.gender
        ages = data_set.age
        num_examples = data_set.shape[0]

        start_time = now()
        total = self.slice[1] - self.slice[0]
        error = 0
        full_tfrecords_path = os.path.join(self.tfrecords_path)
        #if not os.path.exists(full_tfrecords_path):
            #os.makedirs(full_tfrecords_path)
        filename = os.path.join(full_tfrecords_path,'train-%.3d' % self.pid + '.tfrecords')
        print('Writing', filename)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(num_examples):
                if self.need_print(index):
                    ratio = index / total
                    eta = timedelta(seconds=(now() - start_time) * (1 - ratio) / (ratio + 1e-9))
                    print(f'pid: {self.pid}, progress: {round(ratio * 100, 1)}% {index}/{total} images, eta={eta}')
                try:
                    image = cv2.imread(os.path.join(self.dataset_path, str(file_name[index])), cv2.IMREAD_COLOR)
                    image_raw = image.tostring()
                except IOError:  # some files seem not exist in face_data dir
                    error += 1
                    pass
                example = tf.train.Example(features=tf.train.Features(feature={
                    'age': _int64_feature(int(ages[index])),
                    'gender': _int64_feature(int(genders[index])),
                    'image_raw': _bytes_feature(image_raw),
                    'file_name': _bytes_feature(str(file_name[index]).encode('utf-8'))
                    }))
                writer.write(example.SerializeToString())
        print(f'pid: {self.pid}, total: {total} images, time={now() - start_time}')
        print("There are ", error, " missing pictures")
        return total

    @staticmethod
    def need_print(ind):
        if ind == 0:
            return False
        elif ind == 10:
            return True
        elif ind < 1000:
            return ind % 100 == 0
        else:
            return ind % 1000 == 0

    @staticmethod
    def run_job(json_path, dataset_path, tfrecords_path, slice_limits, pid):
        converter = Converter(json_path, dataset_path, tfrecords_path, slice_limits, pid)
        return converter.convert_dataset()

def main():
    n_jobs = 3
    tfrecords_path = '/media/roman/WD1T/tfrecords'
    dataset_path = '/home/roman/data/imdb_wiki'
    json_path = '/home/roman/data/imdb_wiki/train_small.json'
    with open(json_path) as f:
        dataset_len = len(json.load(f))
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = list()
        subsets = np.linspace(0, dataset_len, n_jobs + 1, dtype=np.int)
        for ind in range(n_jobs):
            start = subsets[ind]
            finish = subsets[ind + 1]
            print(f'pid: {ind}, slice: [{start}:{finish}]')
            futures.append(executor.submit(Converter.run_job, json_path, dataset_path, tfrecords_path,
                                           (start, finish), ind))
    total = 0
    for job in as_completed(futures):
        total += job.result()
    print(f'done. Total: {total}')


if __name__ == '__main__':
    main()
