import numpy as np
import tensorflow as tf
import pickle
from glob import glob
import csv
import json
import utils


class SimpleConvModel():
    def __init__(self, params, input, num_filters,
                 num_classes, target_pl, is_training,
                 num_towers, tower_index, graph_models=None):

        self.params = params
        self.batch_size = self.params.batch_size
        self.num_layers = 3
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.num_channels = self.params.num_chan

        # select only the input coresponding to the current tower
        self.input = input[tower_index * self.batch_size * self.params.num_eval_clips:
                           (tower_index+1) * self.batch_size * self.params.num_eval_clips]
        self.target_pl = target_pl[tower_index *
                                   self.batch_size: (tower_index+1) * self.batch_size]

        self.graph_models = graph_models
        self.is_training = is_training

    def add_graph(self, blob_in, graph_model, resize_back, prefix):
        graph_model.set_input(blob_in, prefix=prefix)
        dim_h = blob_in.shape[2]
        dim_w = blob_in.shape[3]

        graph_model.build_model()

        if resize_back:
            graph_out = graph_model.get_regions_feats(dim_h=dim_h, dim_w=dim_w)
        else:
            graph_out = graph_model.get_video_feats()
        return graph_out

    def build_model(self):
        with tf.name_scope('conv_net'):
            with tf.name_scope('conv_net'):
                act = tf.reshape(
                    self.input, [-1, tf.shape(self.input)[2], tf.shape(self.input)[3], self.num_channels])
                for i in range(self.num_layers):
                    with tf.variable_scope('conv' + str(i)):
                        act = tf.layers.conv2d(
                            act, self.num_filters, 3, activation=tf.nn.relu, reuse=False)
                        act = tf.layers.max_pooling2d(act, [2, 2], strides=2)

                output = tf.reduce_mean(act, axis=[1, 2])
                output = tf.reshape(output,
                                    [tf.shape(self.input)[0], tf.shape(self.input)[1],
                                     self.num_filters])

        # graph input is the features map, before the pooling layer
        # reshape to B x TimeSteps x H x W x C
        conv_out = tf.reshape(act,
                              [tf.shape(self.input)[0], tf.shape(self.input)[1],
                               tf.shape(act)[1], tf.shape(
                                  act)[2],
                               self.num_filters])
        self.graph_conv_input = conv_out

        if 'final' in self.params.place_graph:
            graph_out = self.add_graph(
                conv_out, self.graph_models['final'], resize_back=False, prefix='final')
            if self.params.comb_type == 'serial':
                blob_out = tf.expand_dims(tf.expand_dims(
                    tf.expand_dims(graph_out, axis=1), axis=1), axis=1)
            elif self.params.comb_type == 'plus':
                graph_out = tf.expand_dims(tf.expand_dims(
                    tf.expand_dims(graph_out, axis=1), axis=1), axis=1)
                blob_out = blob_out + graph_out

        dropout_i3d = self.params.classification_dropout
        if 'train' in self.params.mode:
            blob_out = tf.layers.dropout(
                blob_out, rate=dropout_i3d, noise_shape=blob_out.shape, training=self.is_training)
        elif 'eval' in self.params.mode:
            blob_out = tf.layers.dropout(
                blob_out, rate=dropout_i3d, training=self.is_training)

        blob_out_test = tf.layers.conv3d(blob_out, filters=self.num_classes,
                                         kernel_size=[1, 1, 1], strides=[1, 1, 1],
                                         padding='VALID',
                                         name='pred')

        # when we evaluate multiple subclips of the same video, the subclips are processed independently,
        # 	a batch has multiple subclips from multiple videos
        # we need to group the subclips from the same video and agregate their result
        if self.params.num_eval_clips > 1:
            blob_out_test = tf.reshape(blob_out_test,
                                       shape=[self.batch_size,
                                              self.params.num_eval_temporal_clips,
                                              self.params.num_eval_spatial_clips,
                                              -1,
                                              tf.shape(blob_out_test)[2],
                                              tf.shape(blob_out_test)[3],
                                              tf.shape(blob_out_test)[4]])
            self.logits = tf.reduce_mean(blob_out_test, axis=[1, 2, 3, 4, 5])
        else:
            self.logits = tf.reduce_mean(blob_out_test, axis=[1, 2, 3])
