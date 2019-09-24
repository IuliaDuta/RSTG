import numpy as np
import tensorflow as tf
import pickle
from glob import glob
import csv
import json
import utils

# This code is a reimplementation in tensorflow of 2D Non-Local model
# of the code available in https://github.com/facebookresearch/video-nonlocal-net
# based on the paper from Wang et. al, CVPR 2018
# "Non-local Neural Networks for Video Classification"


class NLModel_2D():
    def __init__(self, params, input,
                 num_classes, target_pl, is_training,
                 num_towers, tower_index, graph_models=None, i3d=True):

        self.params = params
        self.batch_size = self.params.batch_size
        self.num_classes = num_classes
        self.num_frames = self.params.video_num_frames

        self.target_pl = target_pl[tower_index *
                                   self.batch_size: (tower_index+1) * self.batch_size]
        self.input = input[tower_index * self.batch_size * self.params.num_eval_clips:
                           (tower_index+1) * self.batch_size * self.params.num_eval_clips]

        self.graph_models = graph_models
        self.is_training = is_training

        # because we use small batch_size we do not update statistics 
		# for batch normalization inside the NL Model
        self.is_training_nl = False
        self.i3d = i3d

    def bottleneck_transformation_3d(self, blob_in,  dim_out, stride,
                                     prefix, dim_inner, use_temp_conv=1, temp_stride=1):
        is_training = self.is_training_nl
        blob_in = tf.pad(
            blob_in, [[0, 0], [use_temp_conv, use_temp_conv], [0, 0], [0, 0], [0, 0]])
        blob_out = tf.layers.conv3d(blob_in, filters=dim_inner,
                                    kernel_size=[1 + use_temp_conv * 2, 1, 1],
                                    strides=[temp_stride, 1, 1],
                                    padding='VALID', activation=None,
                                    name=prefix+"_branch2a",
                                    use_bias=False)

        blob_out = tf.layers.batch_normalization(blob_out,
                                                 training=is_training,
                                                 name=prefix+"_branch2a"+'_bn',
                                                 momentum=0.9,
                                                 epsilon=0.00001)
        blob_out = tf.nn.relu(blob_out)

        blob_out = tf.pad(blob_out, [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
        blob_out = tf.layers.conv3d(blob_out, filters=dim_inner,
                                    kernel_size=[1, 3, 3],
                                    strides=[1, stride, stride],
                                    padding='VALID', activation=None,
                                    name=prefix+"_branch2b",
                                    use_bias=False)

        blob_out = tf.layers.batch_normalization(blob_out,
                                                 training=is_training,
                                                 name=prefix+"_branch2b"+'_bn',
                                                 momentum=0.9,
                                                 epsilon=0.00001)
        blob_out = tf.nn.relu(blob_out)

        blob_out = tf.layers.conv3d(blob_out, filters=dim_out,
                                    kernel_size=[1, 1, 1],
                                    strides=[1, 1, 1],
                                    padding='VALID', activation=None,
                                    name=prefix+"_branch2c",
                                    use_bias=False)
        blob_out = tf.layers.batch_normalization(blob_out,
                                                 training=is_training,
                                                 name=prefix+"_branch2c"+'_bn',
                                                 momentum=0.9,
                                                 epsilon=0.00001)
        return blob_out

    def _generic_residual_block_3d(self, blob_in, dim_out, stride, prefix, dim_inner,
                                   group=1, use_temp_conv=0, temp_stride=1):
        tr_blob = self.bottleneck_transformation_3d(blob_in,
                                                    dim_out, stride, prefix, dim_inner,
                                                    use_temp_conv=use_temp_conv,
                                                    temp_stride=temp_stride)
        sc_blob = self._add_shortcut_3d(blob_in, prefix + "_branch1",
                                        dim_out, stride, temp_stride=temp_stride)

        sum_blob = tr_blob + sc_blob
        blob_out = tf.nn.relu(sum_blob)
        return blob_out

    def _add_shortcut_3d(self, blob_in, prefix,
                         dim_out, stride, temp_stride=1):
        is_training = self.is_training_nl
        dim_in = blob_in.shape[-1]
        if dim_in == tf.Dimension(dim_out) and temp_stride == 1 and stride == 1:
            return blob_in
        else:
            blob_in = tf.layers.conv3d(blob_in, filters=dim_out,
                                       kernel_size=[1, 1, 1],
                                       strides=[temp_stride, stride, stride],
                                       padding='VALID',
                                       name=prefix,
                                       use_bias=False)
            blob_in = tf.layers.batch_normalization(blob_in,
                                                    training=is_training,
                                                    name=prefix+'_bn',
                                                    momentum=0.9,
                                                    epsilon=0.00001)
            return blob_in

    def res_stage_nonlocal(self, blob_in,
                           dim_out, stride, num_blocks, prefix,
                           dim_inner=None, use_temp_convs=None, temp_strides=None,
                           batch_size=None, nonlocal_name=None, nonlocal_mod=1000,
                           graph_model=None):

        for idx in range(num_blocks):
            block_prefix = "{}_{}".format(prefix, idx)
            block_stride = 2 if (idx == 0 and stride == 2) else 1
            blob_in = self._generic_residual_block_3d(
                blob_in, dim_out, block_stride, block_prefix,
                dim_inner, use_temp_conv=use_temp_convs[idx], temp_stride=temp_strides[idx])
            dim_in = dim_out

            if idx % nonlocal_mod == nonlocal_mod - 1:
                blob_in = self.add_nonlocal(
                    blob_in, dim_in, dim_in, self.batch_size,
                    nonlocal_name + '_{}'.format(idx), int(dim_in / 2),
                    graph_model=graph_model)
        return blob_in

    def spacetime_nonlocal(self, blob_in, dim_in, dim_out, batch_size, prefix, dim_inner,
                           is_test, max_pool_stride=2):
        cur = blob_in
        # we do projection to convert each spacetime location to a feature
        # theta original size
        # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 4, 14, 14)

        theta = tf.layers.conv3d(cur, filters=dim_inner,
                                 kernel_size=[1, 1, 1],
                                 strides=[1, 1, 1],
                                 padding='VALID', activation=None,
                                 name=prefix+"_theta",
                                 use_bias=True)
        # phi and g: half spatial size
        # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 4, 7, 7)
        max_pool = tf.nn.pool(cur,
                              window_shape=[
                                  1, max_pool_stride, max_pool_stride],
                              pooling_type="MAX",
                              padding="VALID",
                              strides=[1, max_pool_stride, max_pool_stride])

        phi = tf.layers.conv3d(max_pool, filters=dim_inner,
                               kernel_size=[1, 1, 1],
                               strides=[1, 1, 1],
                               padding='VALID', activation=None,
                               name=prefix+"_phi",
                               use_bias=True)
        # phi and g: half spatial size

        g = tf.layers.conv3d(max_pool, filters=dim_inner,
                             kernel_size=[1, 1, 1],
                             strides=[1, 1, 1],
                             padding='VALID', activation=None,
                             name=prefix+"_g",
                             use_bias=True)

        # we have to use explicit batch size (to support arbitrary spacetime size)
        # e.g., (8, 1024, 4, 14, 14) => (8, 1024, 784)

        theta_shape_5d = tf.shape(theta)
        theta = tf.reshape(theta, [batch_size, dim_inner, -1])
        phi = tf.reshape(phi, [batch_size, dim_inner, -1])
        g = tf.reshape(g, [batch_size, dim_inner, -1])

        # e.g., (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi = tf.matmul(theta, phi, transpose_a=True)

        theta_phi_sc = theta_phi * (dim_inner**-.5)

        p = tf.nn.softmax(theta_phi_sc, axis=2)

        # note: g's axis[2] corresponds to p's axis[2]
        # e.g., g(8, 1024, 784_2) * p(8, 784_1, 784_2) => (8, 1024, 784_1)
        t = tf.matmul(g, p, transpose_b=True)

        # reshape back:
        # e.g., (8, 1024, 784) => (8, 1024, 4, 14, 14)
        t_re = tf.reshape(t, shape=theta_shape_5d)

        blob_out = t_re
        blob_out = tf.layers.conv3d(blob_out, filters=dim_out,
                                    kernel_size=[1, 1, 1],
                                    strides=[1, 1, 1],
                                    padding='VALID', activation=None,
                                    name=prefix+"_out",
                                    use_bias=True)

        blob_out = tf.layers.batch_normalization(blob_out,
                                                 training=self.is_training_nl,
                                                 gamma_initializer=tf.zeros_initializer(),
                                                 name=prefix+'_bn',
                                                 momentum=0.9,
                                                 epsilon=0.00001)

        return blob_out

    def add_graph(self, blob_in, graph_model, resize_back):
        graph_model.set_input(blob_in)
        dim_h = blob_in.shape[2]
        dim_w = blob_in.shape[3]

        graph_model.build_model()

        if resize_back:
            graph_out = graph_model.get_regions_feats(dim_h=dim_h, dim_w=dim_w)
        else:
            graph_out = graph_model.get_video_feats()

        graph_out = self.residual_norm(graph_out)
        return graph_out

    def add_nonlocal(self, blob_in, dim_in, dim_out, batch_size, prefix, dim_inner, graph_model=None):
        if graph_model is not None and (self.params.place_graph != 'none'):
            graph_out = self.add_graph(blob_in, graph_model, resize_back=True)
            blob_out = blob_in + graph_out
        else:
            blob_out = self.spacetime_nonlocal(
                blob_in, dim_in, dim_out, batch_size, prefix, dim_inner, self.is_training_nl)
            blob_out = blob_in + blob_out
        return blob_out

    def residual_norm(self, features):
        if self.params.use_norm == 'BN':
            features = tf.layers.batch_normalization(inputs=features, training=self.is_training,
                                                     name='after_graph_bn', gamma_initializer=tf.zeros_initializer())
        elif self.params.use_norm == 'LN':
            features = tf.contrib.layers.layer_norm(inputs=features, scope='after_graph_ln',
                                                    scale=False)
            scale = tf.get_variable(name=f'after_graph_ln_scale', trainable=True,
                                    dtype=tf.float32, shape=features.get_shape()[-1], initializer=tf.zeros_initializer())
            features = scale * features

        return features

    def build_model(self):
        n1, n2, n3, n4 = 3, 4, 6, 3
        dim_inner = 64
        pool_stride = int(self.num_frames/2)
        layer_mod2 = layer_mod3 = layer_mod4 = layer_mod5 = 10000

        # C2D
        use_temp_convs_1 = [0]
        temp_strides_1 = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 6
        temp_strides_4 = [1, ] * 6
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

        with (tf.variable_scope(("backbone"))):
            conv_pad = tf.pad(self.input, [
                              [0, 0], [use_temp_convs_1[0], use_temp_convs_1[0]], [3, 3], [3, 3], [0, 0]])
            conv_blob = tf.layers.conv3d(conv_pad, filters=64,
                                         kernel_size=[1+2*use_temp_convs_1[0], 7, 7], strides=[temp_strides_1[0], 2, 2],
                                         padding='VALID', activation=None,
                                         name='conv1',
                                         use_bias=False)

            bn_blob = tf.layers.batch_normalization(conv_blob,
                                                    training=self.is_training_nl,
                                                    name='res_conv1_bn',
                                                    momentum=0.9,
                                                    epsilon=0.00001)
            relu_blob = tf.nn.relu(bn_blob)

            max_pool = tf.nn.pool(relu_blob,
                                  window_shape=[1, 3, 3],
                                  pooling_type="MAX",
                                  padding="VALID",
                                  strides=[1, 2, 2])

            res_stages = []
            res_stages.append(max_pool)

            if 'res2' in self.params.place_graph:
                layer_mod2 = n1-1

            blob_in = self.res_stage_nonlocal(
                max_pool, 256, stride=1, num_blocks=n1,
                prefix='res2', dim_inner=dim_inner,
                use_temp_convs=use_temp_convs_2, temp_strides=temp_strides_2,
                nonlocal_name='nonlocal_conv2', nonlocal_mod=layer_mod2,
                graph_model=self.graph_models['res2'])
            res_stages.append(blob_in)

            blob_in = tf.nn.pool(blob_in,
                                 window_shape=[2, 1, 1],
                                 pooling_type="MAX",
                                 padding="VALID",
                                 name='pool2',
                                 strides=[2, 1, 1])
            res_stages.append(blob_in)

            # just Non-Local
            if not self.i3d and self.params.place_graph == 'none':
                layer_mod3 = 2
            if 'res3' in self.params.place_graph:
                layer_mod3 = n2-1

            blob_in = self.res_stage_nonlocal(
                blob_in, 512,  stride=2, num_blocks=n2,
                prefix='res3', dim_inner=dim_inner*2,
                use_temp_convs=use_temp_convs_3, temp_strides=temp_strides_3,
                nonlocal_name='nonlocal_conv3', nonlocal_mod=layer_mod3,
                graph_model=self.graph_models['res3'])
            res_stages.append(blob_in)

            # just Non-Local
            if not self.i3d and self.params.place_graph == 'none':
                layer_mod4 = 2
            if 'res4' in self.params.place_graph:
                layer_mod4 = n3-1

            blob_in = self.res_stage_nonlocal(
                blob_in, 1024,  stride=2, num_blocks=n3,
                prefix='res4', dim_inner=dim_inner*4,
                use_temp_convs=use_temp_convs_4, temp_strides=temp_strides_4,
                nonlocal_name='nonlocal_conv4', nonlocal_mod=layer_mod4,
                graph_model=self.graph_models['res4']
            )
            res_stages.append(blob_in)

            if 'res5' in self.params.place_graph:
                layer_mod5 = n4-1
            blob_in = self.res_stage_nonlocal(
                blob_in, 2048,  stride=1, num_blocks=n4,
                prefix='res5', dim_inner=dim_inner*8,
                use_temp_convs=use_temp_convs_5, temp_strides=temp_strides_5,
                nonlocal_name='nonlocal_conv5', nonlocal_mod=layer_mod5,
                graph_model=self.graph_models['res5'])
            res_stages.append(blob_in)

        # blocb_in: shape=(B, time, H, W, 2048)  (3, ?, 8, 8, 2048) (3, ?, 7, 7, 2048)
        # blob out: shape=(B, time, h, w, 2048)  (3, ?, 2, 2, 2048) (3, ?, 1, 1, 2048)
        i3d_out = blob_in

        blob_out = tf.nn.pool(blob_in,
                              # [pool_stride,7,7],
                              window_shape=[pool_stride, 14, 14],
                              pooling_type="AVG",
                              padding="VALID",
                              name='pool5',
                              strides=[1, 1, 1])

        if 'final' in self.params.place_graph:
            graph_out = self.add_graph(
                i3d_out, self.graph_models['final'], resize_back=False)
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
