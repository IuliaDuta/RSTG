import numpy as np
import tensorflow as tf
import pdb
import models.graph_model.gat_layers as gat_layers
from utils.differentiable_resize_area import get_tf_filters, differentiable_resize_area
from models.graph_model.positional_emb import create_map_gauss_embeddings
import utils.utils as utils
import utils.yaml_config as yaml_config

from utils.read_params import read_params
FLAGS = read_params(save=False)


def choose_not_none(a, b):
	return (a if a is not None else b)


class GraphModel():
	def __init__(
			self, params, gc,
			num_nodes, adj_matrix_pl, is_training,
			num_towers, tower_index, graph_name='whole_graph_model', used_num_frames=None,
			backbone_dims=None, project_i3d=None, project_i3d_back=None,
			node_feat_dim=None, message_dims=None):

		self.params = params
		self.graph_name = graph_name
		self.gc = gc
		# when used, multiple subclips are put in a batch
		self.batch_size = self.params.num_eval_clips * params.batch_size
		self.is_training = is_training

		self.num_nodes = num_nodes
		self.number_iterations = self.params.num_process_iter

		self.adj_matrix_pl = adj_matrix_pl
		self.adj_matrix = self.adj_matrix_pl[:, :-1, :-1]

		# use default parameters from params or current instance parameters, when given
		# number of time-steps (frames) of the input of the graph
		self.used_num_frames = choose_not_none(
			used_num_frames, self.params.used_num_frames)
		self.backbone_dims = backbone_dims

		# project the backbone dims to lower dimesion and then project them back after graph processing
		self.project_i3d = project_i3d
		self.project_i3d_back = project_i3d_back

		self.lstm_hid_units = self.params.message_dims
		self.node_feat_dim = choose_not_none(
			node_feat_dim, self.params.node_feat_dim)
		self.message_dims = choose_not_none(
			message_dims, self.params.message_dims)

		self.num_att_heads = 8
		self.att_dim = self.message_dims // self.num_att_heads

		if self.params.use_positional_emb:
			self.dim_map = self.params.dim_map
			self.positional_emb = create_map_gauss_embeddings(
				num_nodes=self.num_nodes, scales=gc.scales, dim_map=self.dim_map)

	# create the nodes from the backbone's feature maps
	def set_input(self, input_feats, prefix=''):
		with tf.variable_scope('graph_input'):
			act = tf.reshape(input_feats, [-1, input_feats.get_shape()[2],
										   input_feats.get_shape()[3], input_feats.get_shape()[4]])

			patch_feats = []
			for scale in self.gc.scales:
				tf_filters_scale = get_tf_filters(act, scale[0], scale[1])
				act_scale = differentiable_resize_area(act, tf_filters_scale)
				act_scale = tf.reshape(
					act_scale, [-1, self.used_num_frames, scale[0]*scale[1], act.get_shape()[3]])
				patch_feats.append(act_scale)
			patch_feats = tf.concat(patch_feats, axis=2)

			self.node_feats = patch_feats
			self.patch_feats = patch_feats

	# project features from graph to features map: RSTG_to_map model
	def remap_nodes(self, dim_h=8, dim_w=8):
		with tf.variable_scope(self.graph_name):
			nodes = self.final_nodes_feats
			nodes = tf.transpose(nodes, [1, 0, 2, 3])

			start = end = out = 0
			for scale in self.gc.scales:
				end += scale[0] * scale[1]
				nodes_scale = tf.reshape(
					nodes[:, :, start:end, :], [-1, scale[0], scale[1], tf.shape(nodes)[3]])
				start += scale[0] * scale[1]
				f_filters_scale_full = get_tf_filters(
					nodes_scale, dim_h, dim_w)
				out_scale = differentiable_resize_area(
					nodes_scale, f_filters_scale_full)
				out = out + out_scale

			out = tf.reshape(
				out, [-1, self.used_num_frames, dim_h, dim_w, tf.shape(out)[3]])
		return out

	# reduce each node dimension
	def project_nodes(self):
		self.node_feats = tf.layers.dense(
			self.node_feats, self.params.message_dims, name='project_i3d')

	# brodcast concat between a: 1 x num_nodes x dim_hid1 si b: 1 x num_nodes x dim_hid2
	#                               => num_nodes x num_nodes x (dim_hid1+dim_hid2)
	def brodcast_concat(self, feats_a, feats_b):
		feats_a = tf.expand_dims(feats_a, axis=1)
		feats_b = tf.expand_dims(feats_b, axis=1)

		feats_bT = tf.transpose(feats_b, [0, 2, 1, 3])
		pairwise = tf.concat(
			[feats_a + tf.zeros_like(feats_bT), tf.zeros_like(feats_a)+feats_bT], axis=3)
		return pairwise

	# returns a 2-layers MLP mesage for every source-destination pair: batch_size x num_nodes x num_nodes x message dimension
	def send_messages_mlp(self, node_feats_pl, name='', reuse=False):
		with tf.variable_scope('send_messages_mlp'):
			if self.params.use_positional_emb:
				positional_emb = tf.tile(tf.expand_dims(
					self.positional_emb, 0), [self.batch_size, 1, 1])
				node_feats_pl = tf.concat([node_feats_pl, positional_emb], 2)

			pairwise_feats = self.brodcast_concat(node_feats_pl, node_feats_pl)

			messages = tf.layers.conv2d(pairwise_feats, self.message_dims, kernel_size=[
										1, 1], activation=tf.nn.relu, name=name+'send_conv1', reuse=reuse)
			messages = tf.layers.conv2d(messages, self.message_dims, kernel_size=[
										1, 1], activation=tf.nn.relu, name=name+'send_conv2', reuse=reuse)

			messages = utils.apply_normalization(
				self.params.use_norm, messages, self.is_training, reuse, name+'message')
			return messages

	# spatial update function
	def update_node_mlp(self, node_feats_pl, agregated_messages, name='update', reuse=False):
		if self.params.scalewise_update == True:
			return self.update_node_mlp_scalewise(node_feats_pl, agregated_messages, name=name, reuse=reuse)
		else:
			return self.update_node_mlp_unique(node_feats_pl, agregated_messages, name=name, reuse=reuse)

	# spatial update when we use the same parameters for each scale (scalewise=False)
	def update_node_mlp_unique(self, node_feats_pl, agregated_messages, name='update', reuse=False):
		with tf.variable_scope('update_node_mlp'):
			node_feats = tf.concat([node_feats_pl, agregated_messages], axis=2)
			updated_nodes = self.update_node_function(
				node_feats, dim=self.node_feat_dim, name=name, suffix='', reuse=reuse)

			updated_nodes = utils.apply_normalization(
				self.params.use_norm, updated_nodes, self.is_training, reuse, name)
			return updated_nodes

	# use different parameters for the update function for each scale (scalewise=True)
	def update_node_mlp_scalewise(self, node_feats_pl, agregated_messages, name='update', reuse=False):
		with tf.variable_scope('update_node_mlp_'):
			node_feats = tf.concat([node_feats_pl, agregated_messages], axis=2)

			tf_mask_type = []
			start = end = 0

			# select just the nodes corresponding to each scale
			for scale in self.gc.scales:
				end += scale[0] * scale[1]
				mask_type = np.array([False] * self.gc.num_nodes)
				mask_type[start:end] = True
				start += scale[0] * scale[1]

				tf_mask_type.append(tf.boolean_mask(
					node_feats, mask_type, axis=1))
			# apply a different function for each scale
			updated_nodes_types = []
			for inc, scale in enumerate(self.gc.scales):
				updated_nodes_types.append(self.update_node_function(
					tf_mask_type[inc], dim=self.message_dims, name=name, suffix=f'_type{inc+1}', reuse=reuse))

			updated_nodes = tf.concat(updated_nodes_types, axis=1)
			updated_nodes = utils.apply_normalization(
				self.params.use_norm, updated_nodes, self.is_training, reuse, name)

			return updated_nodes

	# MLP inside the spatial update function
	def update_node_function(self, node_feats, dim, name, suffix, reuse):
		updated_nodes = node_feats
		for i in range(self.params.update_layers):
			updated_nodes = tf.layers.conv1d(updated_nodes, self.node_feat_dim, kernel_size=1,
											 activation=tf.nn.relu, name=name+'_conv' + str(i+1) + suffix, reuse=reuse)
		return updated_nodes


	# build graph model
	def build_model(self):
		reuse_send = reuse_att = reuse_update = False
		# select where to insert temporal processing stage (before which spatial stage)
		time_iter_mom = [0, 1, 2]

		with tf.variable_scope('time_update_intern'):
			lstm_cell_intern = tf.contrib.cudnn_rnn.CudnnLSTM(
				num_layers=1, num_units=self.lstm_hid_units)

		with tf.variable_scope(self.graph_name):
			if self.project_i3d:
				self.project_nodes()

			if FLAGS.dataset == 'mnist':
				# we only used this extra normalization layer in mnist experiments
				self.node_feats = tf.layers.batch_normalization(
					inputs=self.node_feats, training=self.is_training)

			self.node_feats = utils.apply_normalization(
				self.params.use_norm, self.node_feats, self.is_training, reuse=False, name='before_graf')

			if self.params.node_feat_dim != self.message_dims:
				self.node_feats = tf.layers.conv2d(
					self.node_feats, self.node_feat_dim, 1)
				self.node_feats = utils.apply_normalization(
					self.params.use_norm, self.node_feats, self.is_training, reuse=False, name='before_graf2')

			# batchsize x num_frames x num_nodes x 1024
			crt_spatial_features = tf.transpose(self.node_feats, [1, 0, 2, 3])
			for time_iter in range(self.number_iterations):
				with tf.name_scope('process_graph'):
					# TIME processing stage
					# only for selected stages
					if time_iter in time_iter_mom:
						print('LSTM - Time propagation')
						all_time_processed_nodes = tf.reshape(crt_spatial_features,
															  shape=[self.used_num_frames,
																	 self.batch_size * self.num_nodes,
																	 self.lstm_hid_units])

						lstm_out, _ = lstm_cell_intern(
							all_time_processed_nodes)
						time_node_feats = tf.reshape(
							lstm_out, shape=[self.used_num_frames, self.batch_size, self.num_nodes, -1])
					else:
						time_node_feats = crt_spatial_features

					crt_spatial_features_list = []
					for t_iter in range(self.used_num_frames):
						print('Space propagation')
						crt_node_feats = time_node_feats[t_iter]
						# Send message 
						messages = self.send_messages_mlp(
							crt_node_feats, reuse=reuse_send)

						reuse_send = True
						# Aggregate messages
						if self.params.use_att == 'simple':
							agregated_messages = gat_layers.attn_head(
								crt_node_feats, messages, adj_mat=self.adj_matrix,
								out_sz=self.att_dim, activation=tf.nn.relu, nb_nodes=self.num_nodes,
								reuse=reuse_att, use_norm=self.params.use_norm, is_training=self.is_training)
							reuse_att = True
						elif self.params.use_att == 'dot':
							agregated_messages = gat_layers.dot_attn_head(
								crt_node_feats, messages, adj_mat=self.adj_matrix,
								out_sz=self.att_dim, activation=tf.nn.relu, nb_nodes=self.num_nodes,
								reuse=reuse_att, hid_units=self.params.node_feat_dim, multihead=self.params.multihead_att)
							reuse_att = True

						# Update node
						updated_nodes = self.update_node_mlp(
							crt_node_feats, agregated_messages, reuse=reuse_update)
						reuse_update = True
						crt_node_feats = updated_nodes
						crt_spatial_features_list.append(crt_node_feats)

					crt_spatial_features = tf.stack(
						crt_spatial_features_list, axis=0)

			# FINAL TEMPORAL ITERATION - use different sets of params than previous temporal iteration
			all_time_processed_nodes = tf.reshape(crt_spatial_features,
												  shape=[self.used_num_frames, self.batch_size*self.num_nodes, self.lstm_hid_units])

			with tf.variable_scope('time_update_extern'):
				print('Extern LSTM - time propagation')
				lstm_cell_extern = tf.contrib.cudnn_rnn.CudnnLSTM(
					num_layers=1, num_units=self.lstm_hid_units)
				lstm_out, _ = lstm_cell_extern(all_time_processed_nodes)
			# for prediction use node feats from ALL timesteps (reduce_mean over them)
			# final prediction is obtain as sum over all nodes
			# last time step node features: B x N x C
			self.final_nodes_feats = tf.reshape(
				lstm_out, shape=[self.used_num_frames, self.batch_size, self.num_nodes, -1])

			if self.project_i3d_back:
				# project_nodes_back_to_i3d 
				self.final_nodes_feats = tf.layers.dense(
					self.final_nodes_feats, self.backbone_dims, name='project_back_to_i3d')

			# final video features: B x C
			self.final_video_feats = tf.reduce_sum(
				tf.reduce_mean(self.final_nodes_feats, axis=0), axis=1)

	def get_nodes_feats(self):
		return self.final_nodes_feats

	def get_frames_feats(self):
		return tf.reduce_sum(self.final_nodes_feats, axis=2)

	def get_video_feats(self):
		return self.final_video_feats

	def get_regions_feats(self, dim_h, dim_w):
		return self.remap_nodes(dim_h, dim_w)

	def get_patch_feats(self):
		return self.patch_feats


def main():
	pass
