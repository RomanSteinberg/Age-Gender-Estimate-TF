import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from age_gender.utils.config_parser import get_config
from age_gender.utils.model_saver import ModelSaver
from age_gender.nets.inception_resnet_v1 import InceptionResnetV1
from age_gender.utils.converter import Converter
from tqdm import tqdm


def metric(gt_age, pred_age):
    return abs(gt_age - pred_age)

class Miner:
    def __init__(self, config, converter, sess):
        self.config = config
        self.converter = converter
        self.sess = sess
        preds = None

    def run(self):
        dataset_path = self.config['dataset_path']
        dataset = json.load(Path(dataset_path).open())
        dataset = dataset[:5]
        print(dataset)

        path_to_images = os.path.dirname(dataset_path)

        for item in tqdm(dataset):
            image_path = os.path.join(path_to_images, item['file_name'])
            face = converter.convert_image(image_path)
            faces = np.empty((1, 256, 256, 3))
            faces[0, :, :, :] = face
            pred_age, pred_gender = sess.run([age, gender], feed_dict={images_pl: faces})
            err = metric(item['age'], pred_age[0].tolist())
            item.update({'pred_age': pred_age[0].tolist(), 'pred_gender': pred_gender[0].tolist(), 'err': err})
        Path(config['results_path']).mkdir(exist_ok = True)
        json.dump(dataset, Path(config['results_path']).joinpath('all_results.json').open(mode='w'))
        negative = sorted(dataset, key=lambda row: row['err'], reverse=True)[:self.config['negative_size']]
        json.dump(negative, Path(config['results_path']).joinpath('negative.json').open(mode='w'))
        positive = sorted(dataset, key=lambda row: row['err'], reverse=False)[:self.config['positive_size']]
        json.dump(positive, Path(config['results_path']).joinpath('positive.json').open(mode='w'))

def load_network(config):
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
    config = get_config('config.yaml','mining')
    converter_config = get_config('config.yaml','prepare')
    converter = Converter(converter_config)
    if not config['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess, age, gender, images_pl = load_network(config)
    Miner(config, converter, sess).run()


