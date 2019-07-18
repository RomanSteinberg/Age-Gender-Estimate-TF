import os
import json
import cv2
import tensorflow as tf
from pathlib import Path
from age_gender.utils.dataset_json_loader import DatasetJsonLoader


class DataLoader:
    def __init__(self, dataset_json, data_folder):
        """
        Args:
            data_description_path (string): json description file path.
        """
        self.image_shape = [256, 256, 3]
        self.data_folder = data_folder
        self.description = dataset_json

    def _parse_function(self, sample):
        """
        Parses dict into objects and labels.
        Args:
            sample (dict): object description.
        Returns (tuple):
            Tuple which contains image, age, gender and file name.
        """
        # Convert label from a scalar uint8 tensor to an int32 scalar.
        age = sample['age']
        gender = sample['gender']
        file_path = sample['file_name']
        return age, gender, os.path.join(self.data_folder, file_path)

    def _generator(self):
        for sample in self.description:
            yield self._parse_function(sample)

    def _read_image(self, age, gender, file_path):
        image_string = tf.read_file(file_path)
        image_tn = tf.image.decode_image(image_string, channels=3)
        image_tn.set_shape([None, None, 3])  # important!
        image_tn = tf.reshape(image_tn, self.image_shape)
        image_tn = tf.reverse(image_tn, [-1])
        # image_tn = tf.image.per_image_standardization(image_tn)
        image_tn = tf.math.subtract(tf.math.divide(
            tf.cast(image_tn, dtype=tf.float32), tf.constant(127.5)), tf.constant(1.0))
        return image_tn, age, gender, file_path

    def create_dataset(self, perform_shuffle=False, repeat_count=1, batch_size=1, num_prefetch=None,
                       num_parallel_calls=None):
        """
        Creates tf.data.Dataset object.
        Args:
            perform_shuffle (bool): specifies whether it is necessary to shuffle.
            repeat_count (int): specifies number of dataset repeats.
            batch_size (int): specifies batch size.
        Returns (tuple):
            Tuple which contains images batch and corresponding batches of age labels, gender labels and file names.
        """

        dataset = tf.data.Dataset.from_generator(
            self._generator,
            (tf.int32, tf.int32, tf.string)
        )
        if num_parallel_calls is not None:
            dataset = dataset.map(
                self._read_image, num_parallel_calls=num_parallel_calls)
        else:
            dataset = dataset.map(self._read_image)
        if perform_shuffle:
            # Randomizes input using a window of 256 elements (read into memory)
            dataset = dataset.shuffle(
                buffer_size=256, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        if num_prefetch is not None:
            dataset = dataset.prefetch(num_prefetch)
        return dataset.repeat(repeat_count)

    def dataset_len(self):
        return len(self.description)


def init_data_loader(batch_size, desc_path, images_path, balance_config=None, min_size=None, epochs=None,
                     num_prefetch=None, num_parallel_calls=None):
    print('desc_path', desc_path)
    desc = json.load(Path(desc_path).open())
    if balance_config is not None:
        dataset_json_loader = DatasetJsonLoader(
            balance_config, desc)
        desc = dataset_json_loader.get_dataset()
    loader = DataLoader(desc, images_path)
    repeat_count = 1
    if min_size is not None and loader.dataset_len() < min_size:
        repeat_count = min_size // loader.dataset_len() + (loader.dataset_len() % min_size != 0)
    if epochs is not None:
        repeat_count = epochs
    dataset = loader.create_dataset(
        perform_shuffle=True,
        batch_size=batch_size,
        repeat_count=repeat_count,
        num_prefetch=num_prefetch,
        num_parallel_calls=num_parallel_calls
    )
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    next_data_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)
    return next_data_element, init_op, loader.dataset_len()


def visual_validation(config):
    """
    Функция для валидации чтения с помощью Dataloader. Сохраняет файлы картинок на диск в experiments/test. Важно
    иметь ввиду, что per_image_standardization меняет картинку так, что восстановить и посмотреть ее сложно. Поэтому
    валидацию лучше запускать без испольования Dataloader-ом этого преобразования.
    Args:
        config (dict): конфигурационный файл.
    Returns:
        None
    """
    dataset_path = config['init']['dataset_path']
    face_area_threshold = config['face_area_threshold']
    folder = os.path.join(config['working_dir'], 'experiments/test')
    batch_size = 3
    epochs = 2

    dataset = DataLoader(dataset_path).create_dataset(True, epochs, batch_size)
    iterator = dataset.make_one_shot_iterator()
    all_tn = iterator.get_next()

    with tf.Graph().as_default() and tf.Session() as sess:
        for ep in range(epochs):
            all_data = sess.run(all_tn)
            ep_folder = os.path.join(folder, f'{ep}')
            os.makedirs(ep_folder, exist_ok=True)
            for i, data_piece in enumerate(zip(*all_data)):
                print(data_piece[0].dtype)
                print(i, data_piece[1], data_piece[2], data_piece[3])
                p = os.path.join(folder, f'{ep}/result{i}.jpg')
                cv2.imwrite(p, data_piece[0])
                img = cv2.imread(data_piece[3].decode('utf-8'))
                p = os.path.join(folder, f'{ep}/original{i}.jpg')
                cv2.imwrite(p, img)
