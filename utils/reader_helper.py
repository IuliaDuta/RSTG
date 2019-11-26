import random
import pickle
from glob import glob
import utils.utils as utils
import tensorflow as tf
import numpy as np
import utils.yaml_config as yaml_config
import pdb
from utils.read_params import read_params
FLAGS = read_params(save=False)


def _parse_function_aug(filename, label, vid):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.image.random_brightness(image_decoded, max_delta=0.1)
    y = tf.one_hot(label, FLAGS.num_classes)
    y = tf.expand_dims(y, axis=0)
    return (image_decoded, y, vid)


def _parse_function(filename, label, vid):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    y = tf.one_hot(label, FLAGS.num_classes)
    y = tf.expand_dims(y, axis=0)
    return (image_decoded, y, vid)


def _parse_identity(video, label, vid):
    return (video, label, vid)


class DatasetCreator:
    def __init__(self, num_towers, batch_size):
        self.num_towers = num_towers
        self.batch_size = batch_size

    # data preprocessing + reshape depending on the types of data that we have:
    # multiclip or one clip
    def preprocessing(self, video_pl):
        video_pl_preproc_all = self.inception_preprocessing(video_pl)
        if FLAGS.num_eval_clips > 1:
            video_pl_all = tf.reshape(
                video_pl_preproc_all,
                [self.num_towers * self.batch_size, 1,
                 FLAGS.video_num_frames * 256,
                 256 * FLAGS.num_eval_spatial_clips * FLAGS.num_eval_temporal_clips,
                 FLAGS.num_chan])

            list_video_pl_all = []
            for ii in range(FLAGS.num_eval_temporal_clips):
                for jj in range(FLAGS.num_eval_spatial_clips):
                    list_video_pl_all.append(
                        video_pl_all[:, :, :, 256*(ii*FLAGS.num_eval_spatial_clips+jj):256*(ii*FLAGS.num_eval_spatial_clips+jj+1), :])

            video_pl_all = tf.concat(list_video_pl_all, axis=1)
            video_pl_all = tf.reshape(
                video_pl_all,
                [self.num_towers * self.batch_size * FLAGS.num_eval_temporal_clips * FLAGS.num_eval_spatial_clips,
                 FLAGS.video_num_frames, 256, 256, FLAGS.num_chan])
        else:
            video_pl_all = tf.reshape(
                video_pl_preproc_all,
                [self.num_towers * self.batch_size, -1, FLAGS.dim_h, FLAGS.dim_w, FLAGS.num_chan])
            video_pl_all = video_pl_all[:, :FLAGS.video_num_frames, :, :, :]
            video_pl_all = tf.reshape(
                video_pl_all,
                [self.num_towers * self.batch_size, FLAGS.video_num_frames,
                 FLAGS.dim_h, FLAGS.dim_w, FLAGS.num_chan])
        return video_pl_all

	# create tf.datasets for eval
    def build_dataset(self, images, labels, video_ids, num_towers, sess):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels, video_ids)) \
            .map(self.test_parse_func, num_parallel_calls=4) \
            .batch(self.num_towers * self.batch_size)
        iterator = dataset.make_initializable_iterator()
        handle = sess.run(iterator.string_handle())

        self.data_type = dataset.output_types
        self.data_shape = dataset.output_shapes
        return iterator, handle

	# create tf.datasets for train
    def build_train_dataset(self, images, labels, video_ids, num_towers, sess):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels, video_ids)) \
            .map(self.train_parse_func, num_parallel_calls=4) \
            .shuffle(2*self.num_towers*self.batch_size) \
            .batch(self.num_towers * self.batch_size)
        iterator = dataset.make_initializable_iterator()
        handle = sess.run(iterator.string_handle())

        self.data_type = dataset.output_types
        self.data_shape = dataset.output_shapes
        return iterator, handle

	# choose between test and train dataset creation based on training param
    def create_dataset(self, images, gt_file, num_towers, sess, len_eval=None, name="", training=False):
        images, labels, video_ids = self.read_data(
            images, gt_file, len_eval=len_eval, name=name)
        if training:
            iterator, handle = self.build_train_dataset(
                images, labels, video_ids, self.num_towers, sess)
        else:
            iterator, handle = self.build_dataset(
                images, labels, video_ids, self.num_towers, sess)
        return iterator, handle

    def get_type(self):
        return self.data_type

    def get_shape(self):
        return self.data_shape


class SmtDatasetCreator(DatasetCreator):
    def __init__(self,  num_towers, batch_size):
        super().__init__(num_towers, batch_size)
        self.train_parse_func = _parse_function_aug
        self.test_parse_func = _parse_function
        self.inception_preprocessing = utils.inception_preprocessing_i3d_tf

    def get_video_ids_smt(self, filenames):
        vids = []
        for filename in filenames:
            vids.append(filename.split(
                '/')[-2].split('_')[-1] + '_' + filename.split('/')[-1].split('_')[3])
        return vids

    def get_labels_smt(self, image_paths, gt_dict):
        labels = []
        for i, path in enumerate(image_paths):
            video_name = path.split("/")[-1].split("_")[-3]
            if video_name not in gt_dict:
                print(f'can not find label for {video_name}')
            else:
                label = gt_dict[video_name]['label_id']
                labels.append(label)
        return labels

    def read_data(self, images, gt_file, len_eval=None, name=""):
        print(f'Number videos in {name}: {len(images)}')

        with open(gt_file, 'rb') as fo:
            gt_dict = pickle.load(fo)

        labels = self.get_labels_smt(images, gt_dict)
        video_ids = self.get_video_ids_smt(images)

        return images,  labels, video_ids


class MnistDatasetCreator(DatasetCreator):
    def __init__(self,  num_towers, batch_size):
        super().__init__(num_towers, batch_size)
        self.train_parse_func = _parse_identity
        self.test_parse_func = _parse_identity
        self.inception_preprocessing = utils.inception_preprocessing_mnist

    def read_data(self, images, gt_file, len_eval=None, name=""):
        all_x = all_y = None

        for i, file in enumerate(images):
            with open(file, 'rb') as fo:
                videos_dict = pickle.load(fo)
                x = videos_dict['videos']
                x = np.expand_dims(x, 4)

                y = videos_dict['labels'].astype(int).squeeze()
                y = np.clip(y, 0, FLAGS.num_classes-1)
                y = np.expand_dims(np.eye(FLAGS.num_classes)[y], axis=1)
                if i == 0:
                    all_x = x
                    all_y = y
                else:
                    all_x = np.concatenate((all_x, x), axis=0)
                    all_y = np.concatenate((all_y, y), axis=0)

        if len_eval != None:
            all_x = all_x[:len_eval]
            all_y = all_y[:len_eval]
        video_ids = np.zeros_like(all_y)
        return all_x, all_y, video_ids
