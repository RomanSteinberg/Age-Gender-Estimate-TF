from __future__ import print_function
import argparse
import os
import time
import yaml
import tensorflow as tf
from datetime import datetime

from age_gender.nets.inception_resnet_v1 import inference
from age_gender.utils.queue_reader import inputs, get_file_names
from age_gender.utils.config_parser import get_config

from tensorflow.python import debug as tf_debug

def run_training(config):
    image_path = config['images']
    batch_size = config['batch_size']
    epoch = config['epoch']
    model_path = config['model_path']
    log_dir = config['log_path']
    start_lr = 0.00000001 #config['learning_rate']
    wd = config['weight_decay']
    kp = config['keep_prob']
    epoch_number = 0
    # from tensorflow.python.framework import ops
    # ops.reset_default_graph()
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default() and tf.Session() as sess :
        tf.random.set_random_seed(100)
        check_op = tf.add_check_numerics_ops()
        # Create a session for running operations in the Graph.
        #sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # Input images and labels.
        images, age_labels, gender_labels, _ = inputs(path=get_file_names(image_path), batch_size=batch_size,
                                                      num_epochs=epoch)

        # load network
        # face_resnet = face_resnet_v2_generator(101, 'channels_first')
        train_mode = tf.placeholder(tf.bool)

        age_logits, gender_logits, _ = inference(images, keep_probability=kp,
                                                                     phase_train=train_mode, weight_decay=wd)
        # Build a Graph that computes predictions from the inference model.
        # logits = face_resnet(images, train_mode)

        # if you want to transfer weight from another model,please uncomment below codes
        # sess = restore_from_source(sess,'./models')
        # if you want to transfer weight from another model,please uncomment above codes

        # Add to the Graph the loss calculation.
        bottleneck = [v for v in
                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1/Bottleneck')]
        for var in bottleneck:
            out = tf.verify_tensor_all_finite(var, var.name)
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
        bottleneck = [v for v in
                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1/Bottleneck')]
        #for var in bottleneck:
        #    tf.summary.histogram(var.name, var)

        # Add to the Graph operations that train the model.
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # global_step = tf.train.get_global_step()
        lr = tf.train.exponential_decay(start_lr, global_step=global_step, decay_steps=3000, decay_rate=0.9,
                                        staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        tf.summary.scalar("lr", lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)

        # if you want to transfer weight from another model,please comment below codes
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        # if you want to transfer weight from another model, please comment above codes

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # if you want to transfer weight from another model,please uncomment below codes
        # sess, new_saver = save_to_target(sess,target_path='./models/new/',max_to_keep=100)
        # if you want to transfer weight from another model, please uncomment above codes

        # if you want to transfer weight from another model,please comment below codes
        new_saver = tf.train.Saver(max_to_keep=100)
        pretrained_model_path = os.path.join(model_path,'pretrained_models')
        print('pretrained_model_path',pretrained_model_path)
        ckpt = tf.train.get_checkpoint_state(pretrained_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            new_saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")
        else:
            pass
        # if you want to transfer weight from another model, please comment above codes
        #
        #
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = sess.run(global_step)
            #start_time = time.time()
            #global_start = start_time
            #model_folder = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            #p = os.path.join(model_path, 'trained_models', model_folder)
            #if not os.path.exists(p):
            #    os.makedirs(p)
            while not coord.should_stop():
                # start_time = time.time()
                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.
                _, summary, _ = sess.run([train_op, merged, out], {train_mode: True})
                train_writer.add_summary(summary, step)
                print(f'step: {step-14001}')
                #for var in bottleneck:
                #    out = tf.verify_tensor_all_finite(var, var.name)
                #    with tf.control_dependencies([out]):
                #        sess.run(out)
                #    print_op = tf.print(var, var.name)
                #    with tf.control_dependencies([print_op]):
                #        out = tf.add(tensor, tensor)
                #        sess.run(out)
                # duration = time.time() - start_time
                # # Print an overview fairly often.
                # if step % 100 == 0:
                #     duration = time.time() - start_time
                #     print('%.3f sec' % duration)
                #     start_time = time.time()
                #if (step-14001) % 2882 == 0:
                #    epoch_number += 1
                #    print('epoch_number: ',epoch_number,'time: ',time.time() - start_time)
                #    start_time = time.time()
                #if (step-14001) % 8646 == 0:
                #    save_path = new_saver.save(sess, os.path.join(model_path,'trained_models',model_folder, "model.ckpt"), global_step=global_step)
                #    print("Model saved in file: %s" % save_path)
                #    duration = time.time() - global_start
                #    config['duration'] = duration
                #    json_parameters_path = os.path.join(model_path, 'trained_models', model_folder, "params.yaml")
                #    with open(json_parameters_path, 'w') as file:
                #        yaml.dump(config, file, default_flow_style=False)
                step = sess.run(global_step)
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (epoch, step))
        finally:
            #duration = time.time() - global_start
            #config['duration'] = duration
            #json_parameters_path = os.path.join(model_path,'trained_models',model_folder, "params.yaml")
            #with open(json_parameters_path, 'w') as file:
            #    yaml.dump(config, file, default_flow_style=False)
            # When done, ask the threads to stop.
            #save_path = new_saver.save(sess, os.path.join(model_path,'trained_models',model_folder, "model.ckpt"), global_step=global_step)
            #print("Model saved in file: %s" % save_path)
            coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    config = get_config('config.yaml')['train']
    if not config['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run_training(config)
