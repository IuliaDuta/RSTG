import pdb
import numpy as np
import tensorflow as tf


# attn_head (denoted as 'simple' in config file) is adapted from the code of 
# Graph Attention Network (Velickovic et al., ICLR 2018)

conv1d = tf.layers.conv1d

def attn_head(seq, seq_fts, out_sz, adj_mat, activation, nb_nodes=None,
              in_drop=0.0, coef_drop=0.0, residual=False, reuse=False, use_norm=False, is_training=False, name=''):
    multihead = False
    if multihead:
        no_attention_heads = 8
        all_att_msgs = []
        for at_h in range(no_attention_heads):
            with tf.variable_scope(f'attention_head{at_h}'):
                seq.shape[2]
                seq_h = tf.layers.conv1d(
                    seq, seq.shape[2] // no_attention_heads, 1, name=name+'node_proj', reuse=reuse)
                seq_fts_h = tf.layers.conv2d(seq_fts, seq_fts.shape[3] // no_attention_heads, kernel_size=[
                                             1, 1], name=name+'messages_proj', reuse=reuse)

                att_msg = one_attn_head(seq_h, seq_fts_h, out_sz, adj_mat, activation, nb_nodes,
                                        in_drop, coef_drop, residual, reuse, use_norm, is_training, name)
                all_att_msgs.append(att_msg)

        concat_msgs = tf.concat(all_att_msgs, axis=2)
        out = tf.layers.conv1d(
            concat_msgs, seq.shape[2], 1, name=name+'att_out_proj', reuse=reuse)
        return out
    else:
        return one_attn_head(seq, seq_fts, out_sz, adj_mat, activation, nb_nodes,
                             in_drop, coef_drop, residual, reuse, use_norm, is_training, name)


def one_attn_head(seq, seq_fts, out_sz, adj_mat, activation, nb_nodes=None,
                  in_drop=0.0, coef_drop=0.0, residual=False, reuse=False, use_norm=False, is_training=False, name=''):
    # attention based on a simple sum of features (projected using 2 diff ws)
    # alpha(i,j) = <(W1*f_i) + (W2*f_j^T)>
    with tf.variable_scope('my_attn'):
        # if in_drop != 0.0:
        #    seq = tf.nn.dropout(seq, 1.0 - in_drop)
        #seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq, 1, 1, name='att_conv1', reuse=reuse)
        f_2 = tf.layers.conv1d(seq, 1, 1, name='att_conv2', reuse=reuse)

        f_1 = tf.multiply(adj_mat, f_1)
        f_2 = tf.multiply(adj_mat, tf.transpose(f_2, [0, 2, 1]))

        logits = f_1 + f_2
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + adj_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        coefs = tf.expand_dims(coefs, axis=3)
        vals = tf.multiply(coefs, seq_fts)
        vals = tf.reduce_sum(vals, axis=2)

        ret = tf.contrib.layers.bias_add(vals, reuse=reuse, scope='att_bias')

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1, name='att_conv_res')
            else:
                seq_fts = ret + seq

        if use_norm == 'BN':
            ret = tf.layers.batch_normalization(
                inputs=ret, training=is_training, reuse=reuse, name='att_bn')
        elif use_norm == 'LN':
            ret = tf.contrib.layers.layer_norm(
                inputs=ret, reuse=reuse, scope='att_ln')

        return activation(ret)


def dot_attn_head(seq, seq_fts, out_sz, adj_mat, activation, nb_nodes=None, in_drop=0.0,
                  coef_drop=0.0, residual=False, reuse=False, hid_units=512, multihead=False):
    multihead = multihead
    if multihead:
        no_attention_heads = 8
        all_att_msgs = []
        for at_h in range(no_attention_heads):
            with tf.variable_scope(f'attention_head{at_h}'):
                seq_fts_h = tf.layers.conv2d(
                    seq_fts, seq_fts.shape[3] // no_attention_heads, kernel_size=[1, 1], name='messages_proj', reuse=reuse)

                att_msg = one_dot_attn_head(seq, seq_fts_h, out_sz, adj_mat, activation, nb_nodes,
                                            in_drop, coef_drop, residual, reuse, hid_units // no_attention_heads)
                all_att_msgs.append(att_msg)

        concat_msgs = tf.concat(all_att_msgs, axis=2)
        return concat_msgs

    else:
        return one_dot_attn_head(seq, seq_fts, out_sz, adj_mat, activation, nb_nodes,
                                 in_drop, coef_drop, residual, reuse, hid_units)


def one_dot_attn_head(seq, seq_fts, out_sz, adj_mat, activation, nb_nodes=None, in_drop=0.0,
                      coef_drop=0.0, residual=False, reuse=False, hid_units=512):
    # attention based on dot products between features
    # alpha(i,j) = <(W1*f_i), (W2*f_j^T)>
    with tf.variable_scope('dot_product_attn'):
        f_1 = tf.layers.conv1d(
            seq, hid_units, 1, name='att_conv1', reuse=reuse)
        f_2 = tf.layers.conv1d(
            seq, hid_units, 1, name='att_conv2', reuse=reuse)

        f_1 = tf.expand_dims(f_1, 2)
        f_2 = tf.expand_dims(f_2, 1)
        adj_mat = tf.expand_dims(adj_mat, 3)
        logits = tf.multiply(adj_mat, f_1 * f_2)
        att_logits = tf.reduce_sum(logits, axis=3)

        coefs = tf.nn.softmax(att_logits)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        coefs = tf.expand_dims(coefs, axis=3)
        vals = tf.multiply(coefs, seq_fts)
        vals = tf.reduce_sum(vals, axis=2)

        ret = tf.contrib.layers.bias_add(vals, reuse=reuse, scope='att_bias')

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1, name='att_conv_res')
            else:
                seq_fts = ret + seq

        return activation(ret)
