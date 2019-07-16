import tensorflow as tf
import os
from age_gender.utils.config_parser import get_config
from age_gender.utils.dataloader import init_data_loader


def validate(config):
    validation_config = get_config(config, 'dataset_validation')
    dataset_config = get_config(config, 'datasets')[
        validation_config['dataset']]
    batch_size = validation_config['batch']
    num_epochs = validation_config['epochs']
    next_data_element, train_init_op, train_size = init_data_loader(
        batch_size,
        dataset_config['full_desc_path'],
        dataset_config['images_path'],
        dataset_config['balance'], epochs=num_epochs
    )
    print('dataset_size: ', train_size)
    print('train_size // batch_size', train_size // batch_size)
    num_batches = train_size // batch_size + \
        (train_size % batch_size != 0)
    with tf.Graph().as_default() and tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('num_epochs*num_batches', num_epochs*num_batches)
        sess.run(train_init_op)
        for batch_idx in range(num_epochs*num_batches):
            train_images, train_age_labels, train_gender_labels, file_paths = sess.run(
                next_data_element)
            print(f'batch_inx: {batch_idx}, file_paths_len: {len(file_paths)}')


if __name__ == '__main__':
    config = get_config('config.yaml')
    if not config['dataset_validation']['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    validate(config)
