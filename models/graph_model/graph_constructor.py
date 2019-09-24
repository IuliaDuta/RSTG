import tensorflow as tf
import numpy as np

import pdb
import utils

from glob import glob
from tensorflow import logging


class PatchConstructor():
    def __init__(self, params):
        self.params = params
        self.dim_h = self.params.dim_h
        self.dim_w = self.params.dim_w

        self.scales = [(i, i) for i in range(self.params.num_scales, 0, -1)]
        self.num_nodes = sum([(h*w) for (h, w) in self.scales])
        self.all_patches_coord = self.generate_all_patches_coord()

    def crop_one_patch(self, videos, x0, x1, y0, y1):
        # crop a patch (volume) by coordinates
        # 		videos_size: batch_size x num_frames x dim_h x dim_w x 1
        # 		patch_size: batch_size x num_frames x dim_ph x dim_pw x 1
        return videos[:, :, y0:y1, x0:x1, :]

    def intersection(self, patch1, patch2):
        # check if 2 rectangles intersect each others
        # 		patch1, patch2:  coordinates of rectangles
        # 		out: a boolean true/false

        x0, x1, y0, y1 = patch1
        a0, a1, b0, b1 = patch2

        min_x = max(x0, a0)
        max_x = min(x1, a1)
        min_y = max(y0, b0)
        max_y = min(y1, b1)

        # case when the recangles intersect in a single point (diagonal neigh)
        if (((max_x - min_x) == 0) and ((max_y - min_y) == 0)):
            return False
        return (((max_x - min_x) >= 0) and ((max_y - min_y) >= 0))

    def get_adjacency(self):
        # compute adjacency matrix based on patch proposals
        patches_coord_flat = np.concatenate(self.all_patches_coord)
        num_nodes = patches_coord_flat.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i):
                if self.intersection(patches_coord_flat[i], patches_coord_flat[j]):
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1

        # add one extra node
        adj_new = np.zeros((adj_matrix.shape[0]+1, adj_matrix.shape[1]+1))
        adj_new[:-1, :-1] = adj_matrix
        adj_new[:adj_matrix.shape[0], adj_matrix.shape[1]] = 1
        adj_new[adj_matrix.shape[0], :adj_matrix.shape[1]] = 1

        adj_matrix = np.expand_dims(adj_new, 0)
        adj_matrix = np.tile(adj_matrix, (self.params.batch_size *
                                          self.params.num_eval_temporal_clips * self.params.num_eval_spatial_clips, 1, 1))

        return adj_matrix, self.num_nodes

    # scale = num of disjoint patch per axis (scale=3 => split in 3x3)
    def extract_scalewise_patches_coord(self, scale_h, scale_w, stride_y=None, stride_x=None, ):
        # scale_h, scale_w: scale of sliding window
        # stride_x, stride_h: stride for sliding window (default is without overlap)
        # 		=> [(x0,x1,y0,y1),..]

        h_patch = int(self.dim_h / scale_h)
        w_patch = int(self.dim_w / scale_w)

        if stride_y is None:
            stride_y = h_patch
        if stride_x is None:
            stride_x = w_patch

        y0 = 0
        y1 = y0 + h_patch

        max_x = self.dim_w
        max_y = self.dim_h

        patches_coord = []
        while(y1 <= max_y):
            x0 = 0
            x1 = x0 + w_patch
            while(x1 <= max_x):
                patch_coord = [x0, x1, y0, y1]
                patches_coord.append(patch_coord)
                x0 += stride_x
                x1 = x0 + w_patch
            y0 += stride_y
            y1 = y0 + h_patch

        print(scale_h, scale_w, h_patch, w_patch, len(patches_coord))
        return patches_coord

    def generate_all_patches_coord(self):
        # apply proposal coords for each scales
        # 		=> num_scales lists of [(x0,x1,y0,y1),..]
        # one for all videos in dataset

        all_patches_coord = []
        for (sh, sw) in self.scales:
            patches_coord = self.extract_scalewise_patches_coord(sh, sw)
            all_patches_coord.append(patches_coord)
        return all_patches_coord
