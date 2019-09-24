import numpy as np
import random
import pdb
from glob import glob
import pickle
import tensorflow as tf


def apply_normalization(norm_type, norm_input, is_training, reuse, name):
    norm_output = norm_input
    if norm_type == 'BN':
        norm_output = tf.layers.batch_normalization(
            inputs=norm_input, training=is_training, reuse=reuse, name=name + "_bn")
    elif norm_type == 'LN':
        norm_output = tf.contrib.layers.layer_norm(
            inputs=norm_input, reuse=reuse, scope=name + "_ln")
    return norm_output


def count_params(variables, constraint):
    total_params = 0
    for v in variables:
        prod = 1
        if v.shape.dims is not None:
            for d in v.shape.dims:
                prod *= d
        if constraint in v.name:
            total_params += prod

    print(constraint + f' NUMBER OF PARAMS: {total_params}')
    return total_params


def inception_preprocessing_mnist(images):
    return tf.cast(images, tf.float32)


def inception_preprocessing_i3d_tf(images):
    images = tf.cast(images, tf.float32)
    images = tf.subtract(images, 114.75)
    images = tf.div(images, 57.375)
    return images


def read_data_mnist(file, num_classes=65):
    with open(file, 'rb') as fo:
        videos_dict = pickle.load(fo)
        x = np.expand_dims(videos_dict['videos'], 4)
        y = videos_dict['labels'].astype(int).squeeze()
        y = np.clip(y, 0, num_classes-1)
        y = np.expand_dims(np.eye(num_classes)[y], axis=1)
    return x, y
