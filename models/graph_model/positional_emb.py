import numpy as np
import pdb
import tensorflow as tf

from scipy import signal


def gkern(kernlen=21, std=3):
    # Returns a 2D Gaussian kernel array
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def create_map_gauss_embeddings(num_nodes, scales, dim_map):
    pos_emb = np.zeros((num_nodes, dim_map*dim_map))
    idx_node = 0
    for scale in scales:
        h = dim_map // scale[0]
        w = dim_map // scale[1]
        gauss_kernel = gkern(h, 2)
        for i in range(scale[0]):
            for j in range(scale[1]):
                up_left = (i*w, j*h)
                down_right = ((i+1)*w, (j+1)*h)
                crt_map = np.zeros((dim_map, dim_map))
                crt_map[up_left[0]:down_right[0], up_left[1]
                    :down_right[1]] = gauss_kernel
                pos_emb[idx_node] = crt_map.flatten()
                idx_node = idx_node + 1

    positional_emb = tf.get_variable(name=f'positional_emb', trainable=False, dtype=tf.float32,
                                     shape=[num_nodes, dim_map*dim_map],
                                     initializer=tf.constant_initializer(pos_emb))
    return positional_emb
