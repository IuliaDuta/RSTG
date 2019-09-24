import tensorflow as tf
import pickle
import os
import numpy as np
# the Tensorflow implementation of tf.image.resize_area does not have defined gradient
# so we implement it ourselfs


def get_resize_filters(H, W, h, w):
    saved_filters_name = f'./util_pickles/image_resize_area_align_false_filters_HWhw_{H}_{W}_{h}_{w}.pickle'
    if os.path.exists(saved_filters_name):
        with open(saved_filters_name, 'rb') as handle:
            saved_filters = pickle.load(handle)
        filters = saved_filters['filters']
    else:
        print(
            f"Filters of dimension {h, w, H, W} do not exists in util_pickles: Create filters.")
        with tf.device('cpu'):
            images = tf.placeholder(tf.float32, [1, H, W, 1])
            img_resized = tf.image.resize_area(
                images, [h, w], align_corners=False)

            filters = np.zeros([h, w, H, W])
            for i in range(H):
                for j in range(W):
                    images_val = np.zeros([H, W])
                    images_val[i, j] = 1
                    images_val = np.expand_dims(images_val, 0)
                    images_val = np.expand_dims(images_val, 3)

                    feed_dict = {
                        images: images_val
                    }

                    sess = tf.Session()
                    img_resized_val = sess.run(
                        img_resized, feed_dict=feed_dict)
                    out = img_resized_val[0, :, :, 0]
                    filters[:, :, i, j] = out
        with open(saved_filters_name, 'wb') as handle:
            pickle.dump({'filters': filters}, handle)
    return filters


def get_tf_filters(image, h, w, H=None, W=None):
    if H is None:
        H = image.get_shape()[1]
        W = image.get_shape()[2]
    filters = get_resize_filters(H, W, h, w)
    with tf.variable_scope('resize', reuse=tf.AUTO_REUSE):
        tf_filters = tf.get_variable(name=f'resize_filters__{H}_{W}_{h}x{w}', trainable=False, dtype=tf.float32,
                                     shape=[h, w, H, W],
                                     initializer=tf.constant_initializer(filters))
    return tf_filters


def get_tf_filters_pl(image, h, w):
    H = image.get_shape()[1]
    W = image.get_shape()[2]
    filters = get_resize_filters(H, W, h, w)
    filters_pl = tf.placeholder(
        tf.float32, [h, w, H, W], name=f'resize_filter_{h}x{w}')

    return filters_pl, filters.astype(np.float32)


def differentiable_resize_area(images, tf_filters):
    B = images.get_shape()[0]
    H = images.get_shape()[1]
    W = images.get_shape()[2]
    C = images.get_shape()[3]
    h = tf_filters.get_shape()[0]
    w = tf_filters.get_shape()[1]

    tf_filters_rep = tf.tile(tf.expand_dims(tf_filters, 0), [B, 1, 1, 1, 1])

    tf_filters_res = tf.reshape(tf_filters_rep, [B, h*w, H*W])
    inp_res = tf.reshape(images, [B, H*W, C])

    my_tf_resize = tf.matmul(tf_filters_res, inp_res)
    my_tf_resize = tf.reshape(my_tf_resize, [B, h, w, C])
    return my_tf_resize


def differentiable_resize_area_v2(images, tf_filters):
    B = images.get_shape()[0]
    H = images.get_shape()[1]
    W = images.get_shape()[2]
    C = images.get_shape()[3]
    h = tf_filters.get_shape()[0]
    w = tf_filters.get_shape()[1]

    inp_res = tf.reshape(images, [B, 1, 1, H, W, C])
    tf_filters_res = tf.reshape(tf_filters, [1, h, w, H, W, 1])
    my_tf_resize = inp_res * tf_filters_res
    my_tf_resize = tf.reduce_sum(my_tf_resize, [3, 4])
    return my_tf_resize
