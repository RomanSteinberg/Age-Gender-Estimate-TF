import argparse
import os
from age_gender.utils.model_saver import ModelSaver
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
from age_gender.nets.inception_resnet_v1 import InceptionResnetV1


class HistogramWriter:
    def __init__(self, models_dir):
        self.model = InceptionResnetV1()
        self.images = tf.placeholder(
            tf.float32, shape=[None, 256, 256, 3])
        self.global_step = self.model.global_step
        self.models_dir = models_dir
        self.log_dir = os.path.join('debug', self.models_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def get_histogram(self, bottleneck, summary_writer, global_step):
        for var in bottleneck:
            summary = summary_pb2.Summary()
            hist = log_histogram(var.name, var.eval())
            summary.value.add(tag=var.name, histo=hist)
            summary_writer.add_summary(summary, global_step)

    def run(self):
        summary_writer = tf.summary.FileWriter(self.log_dir)
        with tf.Session() as sess:
            for model_checkpoint in tqdm(glob("experiments/"+self.models_dir+"/model.ckpt-*.meta")):
                path = model_checkpoint.replace('.meta', '')
                var_to_restore, _, _ = self.model.inference(self.images)
                saver = ModelSaver(var_to_restore)
                saver.restore_model(sess, path)
                trained_steps = sess.run(self.global_step)
                print('trained_steps', trained_steps)
                init = tf.initialize_all_variables()
                sess.run(init)
                bottleneck = [v for v in tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model.bottleneck_scope)]
                self.get_histogram(bottleneck, summary_writer, trained_steps)
                tf.get_variable_scope().reuse_variables()


def log_histogram(tag, values, bins=1000):
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    #writer.add_summary(summary, step)
    # writer.flush()
    return hist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, help="path to directory with saved models")
    args = parser.parse_args()
    HistogramWriter(args.models_dir).run()
