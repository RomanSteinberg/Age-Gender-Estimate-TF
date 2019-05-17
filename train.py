import os
import yaml
import tensorflow as tf
from datetime import datetime, timedelta

from age_gender.nets import inception_resnet_v1
from age_gender.utils.dataloader import DataLoader
from age_gender.utils.config_parser import get_config


def run_training(config):
    working_dir = config['working_dir']
    num_epochs = config['epoch']
    batch_size = config['batch_size']
    save_frequency = config['init']['save_frequency']
    pretrained_model_folder = os.path.join(working_dir, 'models/pretrained_models')
    experiment_folder = os.path.join(working_dir, 'experiments', datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))
    os.makedirs(experiment_folder, exist_ok=True)
    log_dir = os.path.join(experiment_folder, 'logs')

    dataset_size, global_step, train_mode, init_op, train_op, reset_global_step_op = create_computational_graph(config)
    num_batches = (dataset_size + 1) // batch_size
    print(f'Dataset size: {dataset_size}, epochs in train: {num_epochs}, batches in epoch: {num_batches}')

    with tf.Graph().as_default() and tf.Session() as sess:
        sess.run(init_op)
        summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # if you want to transfer weight from another model,please uncomment below codes
        # sess, new_saver = save_to_target(sess,target_path='./models/new/',max_to_keep=100)
        # if you want to transfer weight from another model, please uncomment above codes

        saver = tf.train.Saver(max_to_keep=100)
        ckpt = tf.train.get_checkpoint_state(pretrained_model_folder)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(reset_global_step_op)
            print('Pretrained model loaded')
        # todo: нужен код продолжения обучения модели, при этом номер эпохи должен начинаться не с 1

        start_time = {'train': datetime.now()}

        for epoch in range(1, num_epochs+1):
            start_time.update({'epoch': datetime.now()})
            for batch_idx in range(num_batches):
                _, summary, step = sess.run([train_op, summaries, global_step], {train_mode: True})
                train_writer.add_summary(summary, step)

            t = time_spent(start_time['epoch'])
            print(f'Epoch {epoch} spent {t}')
            if epoch % save_frequency == 0 or epoch == 1:
                save_path = saver.save(sess, os.path.join(experiment_folder, "model.ckpt"), global_step=epoch)
                save_hyperparameters(config, experiment_folder, start_time)
                print("Model saved in file: %s" % save_path)

        save_path = saver.save(sess, os.path.join(experiment_folder, "model.ckpt"), global_step=epoch)
        save_hyperparameters(config, experiment_folder, start_time)
        print("Model saved in file: %s" % save_path)


def save_hyperparameters(config, experiment_folder, start_time):
    config['duration'] = time_spent(start_time['train'])
    json_parameters_path = os.path.join(experiment_folder, "hyperparams.yaml")
    with open(json_parameters_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def create_computational_graph(config):
    dataset_path = config['init']['dataset_path']
    face_area_threshold = config['face_area_threshold']
    start_lr = config['learning_rate']
    wd = config['weight_decay']
    kp = config['keep_prob']
    batch_size = config['batch_size']
    epochs = config['epoch']

    loader = DataLoader(dataset_path, face_area_threshold)
    dataset = loader.create_dataset(True, epochs, batch_size)
    dataset_len = loader.dataset_len()
    iterator = dataset.make_one_shot_iterator()
    images, age_labels, gender_labels, im_paths = iterator.get_next()

    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = inception_resnet_v1.inference(images, keep_probability=kp,
                                                                 phase_train=train_mode, weight_decay=wd)

    # head
    age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits)
    age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)
    gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels,
                                                                          logits=gender_logits)
    gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

    # l2 regularization
    total_loss = tf.add_n(
        [gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    abs_loss = tf.losses.absolute_difference(age_labels, age)
    gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))
    tf.summary.scalar("age_cross_entropy", age_cross_entropy_mean)
    tf.summary.scalar("gender_cross_entropy", gender_cross_entropy_mean)
    tf.summary.scalar("total loss", total_loss)
    tf.summary.scalar("train_abs_age_error", abs_loss)
    tf.summary.scalar("gender_accuracy", gender_acc)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    reset_global_step_op = tf.assign(global_step, 0)
    lr = tf.train.exponential_decay(start_lr, global_step=global_step, decay_steps=3000, decay_rate=0.9, staircase=True)
    optimizer = tf.train.AdamOptimizer(lr)
    tf.summary.scalar("lr", lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return dataset_len, global_step, train_mode, init_op, train_op, reset_global_step_op


def time_spent(start):
    sec = int((datetime.now() - start).total_seconds())
    return str(timedelta(seconds=sec))


if __name__ == '__main__':
    config = get_config('config.yaml', 'train')
    if not config['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run_training(config)
