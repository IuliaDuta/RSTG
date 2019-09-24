from __future__ import absolute_import
from utils.read_params import read_params
from utils.reader_helper import SmtDatasetCreator, MnistDatasetCreator
from models.backbone_models.Simple_baseline_graph import SimpleConvModel
from models.backbone_models.C2D_NL_baseline_graph_50 import NLModel_2D
from models.backbone_models.NL_baseline_graph_50 import NLModel
from models.graph_model.graph_model import GraphModel
import tensorflow.contrib.slim as slim
from tensorflow.python.client import device_lib
from utils.inspect_checkpoint2 import print_tensors_in_checkpoint_file2
from models.graph_model.graph_constructor import PatchConstructor
import csv
import pickle
import json
import pylab
from glob import glob
import pdb
import random
import utils.utils as utils
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..')))
print(sys.path)


FLAGS = read_params()
if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)


def initialize(sess, latest_checkpoint):
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    # Count and print number of parameters in model
    _ = utils.count_params(all_variables, constraint='')
    _ = utils.count_params(all_variables, constraint='backbone')
    _ = utils.count_params(all_variables, constraint='graph')

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    if FLAGS.restore:
        checkpoint = tf.train.latest_checkpoint(FLAGS.restore_dir)
        print('Restore from ' + checkpoint)
        print_tensors_in_checkpoint_file2(
            file_name=checkpoint, tensor_name='',
            all_tensors=False, all_tensor_names=True)
        # do not restore the filters used in resize_area
        restore_vars = [v for v in  all_variables if 'resize_filters' not in v.name]
        print('Restore variables: ')
        [print(v) for v in restore_vars]

        restore_saver = tf.train.Saver(
            restore_vars, max_to_keep=100000, reshape=True)
        restore_saver.restore(sess,  checkpoint)

def save_model(saver, model_dir, sess, iter='last'):
    directory = model_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = saver.save(sess, directory + "/model_"+str(iter)+".ckpt")
    print("Model saved in file: %s" % save_path)


def eval_model(sess, backbone_model, graph_model, merged_summaries, valid_writer,
               global_step_val, eval_mean_acc_summary, adj_matrix,
               len_test=512, num_towers=1,
               tower_accuracy=-1, tower_accuracy_top_k=-1,
               tower_loss=-1, handle_pl=None,
               handle_val=None, name='test'):

    num_it = int(len_test / (FLAGS.batch_size * num_towers))
    mean_loss_ = mean_acc_ = mean_acc_top_k_ = 0
    rand_i = random.randint(0, num_it-1)

    feed_dict = {
        backbone_model.is_training: False,
        handle_pl: handle_val
    }

    if graph_model:
        feed_dict.update({
            graph_model.adj_matrix_pl: adj_matrix,
            graph_model.is_training: False
        })

    for i in range(num_it):
        [summaries_eval_val, loss_eval_val,
         acc_eval_val, acc_top_k_eval_val] = sess.run([
             merged_summaries, tower_loss,
             tower_accuracy, tower_accuracy_top_k
         ], feed_dict=feed_dict)

        mean_loss_ = mean_loss_ + np.array(loss_eval_val).sum()
        mean_acc_ = mean_acc_ + np.array(acc_eval_val).sum()
        mean_acc_top_k_ = mean_acc_top_k_ + np.array(acc_top_k_eval_val).sum()

        print((f'[{name}] intermediate ({(i+1)*FLAGS.batch_size*num_towers} examples) '
               f'mean accuracy : {mean_acc_ / ((i+1) * num_towers)} '
               f'mean top-5 accuracy : {mean_acc_top_k_ / ((i+1)  * num_towers)}'))
        if i == rand_i:
            summaries_eval_shown_val = summaries_eval_val

    eval_loss = mean_loss_ / num_it
    eval_acc = mean_acc_ / (num_it * num_towers)
    eval_acc_top_k = mean_acc_top_k_ / (num_it * num_towers)
    print((f'[{name}][{(i+1)*FLAGS.batch_size*num_towers} examples] '
           f'EVAL Step loss is:  {eval_loss} '
           f'acc is: {eval_acc} top-5 acc is: {eval_acc_top_k}'))

    sys.stdout.flush()

    if FLAGS.mode == 'train':
        eval_mean_acc_summary.value[0].simple_value = eval_acc
        valid_writer.add_summary(eval_mean_acc_summary, global_step_val)
        valid_writer.add_summary(summaries_eval_shown_val, global_step_val)

    return eval_acc


def combine_gradients(tower_grads):
    filtered_grads = [[x for x in grad_list if x[0] is not None]
                      for grad_list in tower_grads]
    final_grads = []
    for i in range(len(filtered_grads[0])):
        grads = [filtered_grads[t][i] for t in range(len(filtered_grads))]
        grad = tf.stack([x[0] for x in grads], 0)
        grad = tf.reduce_sum(grad, 0, name=grads[0][1].name[:-2])
        final_grads.append((grad, filtered_grads[0][i][1],))

    return final_grads


def compute_loss(model):
    with tf.name_scope('accuracy'):
        correct_pred = tf.cast(
            tf.equal(
                tf.argmax(model.logits, 1),
                tf.argmax(model.target_pl[:, 0, :], 1)
            ), tf.float32)
        accuracy = tf.reduce_mean(correct_pred)
    topk = 5
    with tf.name_scope(f'top-{topk}-accuracy'):
        gt_labels = tf.argmax(model.target_pl[:, 0, :], 1)
        _, top_k_ind = tf.math.top_k(model.logits, k=topk)
        all_topk = []
        for kk in range(topk):
            correct_pred = tf.cast(
                tf.equal(
                    tf.cast(top_k_ind[:, kk], tf.int64), gt_labels
                ), tf.float32)
            all_topk.append(correct_pred)
        correct_pred = tf.stack(all_topk)
        correct_pred = tf.reduce_max(correct_pred, axis=0)
        accuracy_top_k = tf.reduce_mean(correct_pred)

    # loss
    crossent = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=model.target_pl[:, 0, :], logits=model.logits)

    return crossent, accuracy, accuracy_top_k


def create_models(num_towers, tower_i, video_pl_all_tower, target_pl_all_tower, adj_matrix_pl, is_training_pl, gc):
    graph_models = {}
    all_flags_graph = {
        'res2': {'backbone_dims': 256, 'project_i3d': True, 'project_i3d_back': True, 'used_num_frames': FLAGS.video_num_frames},
        'res3': {'backbone_dims': 512, 'project_i3d': False, 'project_i3d_back': False, 'used_num_frames': FLAGS.used_num_frames},
        'res4': {'backbone_dims': 1024, 'project_i3d': True, 'project_i3d_back': True, 'used_num_frames': FLAGS.used_num_frames},
        'res5': {'backbone_dims': 2048, 'project_i3d': True, 'project_i3d_back': True, 'used_num_frames': FLAGS.used_num_frames},
        'final': {'backbone_dims': 2048, 'project_i3d': True, 'project_i3d_back': True, 'used_num_frames': FLAGS.used_num_frames}
    }
    if FLAGS.dataset == 'mnist':
        all_flags_graph['final'] = {'backbone_dims': 2048, 'project_i3d': False,
                                    'project_i3d_back': False, 'used_num_frames': FLAGS.used_num_frames}

    flags_graph = {}
    for key, val in all_flags_graph.items():
        flags_graph[key] = None
        if key in FLAGS.place_graph:
            flags_graph[key] = val

    for place_name, flags_gr in flags_graph.items():
        graph_models[place_name] = None
        if place_name in FLAGS.place_graph:
            # create a graph model for each place in the architcture denoted by "place_name" var
            # all graphs shared the same placeholder for
            #       adjacency matrix; traget; is_training
            current_graph_model = GraphModel(FLAGS, gc, gc.num_nodes,
                                             adj_matrix_pl=adj_matrix_pl, is_training=is_training_pl,
                                             num_towers=num_towers,
                                             tower_index=tower_i,
                                             used_num_frames=flags_gr['used_num_frames'],
                                             backbone_dims=flags_gr['backbone_dims'],
                                             project_i3d=flags_gr['project_i3d'],
                                             project_i3d_back=flags_gr['project_i3d_back'],
                                             graph_name=f'whole_graph_model_{place_name}')
            graph_models[place_name] = current_graph_model

    # create the backbone combined with the RSTG model
    if FLAGS.backbone_model == 'i3d':
        # Resnet-50 I3D model
        with tf.variable_scope(("resnet_i3d")):
            backbone_model = NLModel(FLAGS, video_pl_all_tower,
                                     FLAGS.num_classes,
                                     target_pl=target_pl_all_tower,
                                     is_training=is_training_pl,
                                     num_towers=num_towers,
                                     tower_index=tower_i,
                                     graph_models=graph_models,
                                     i3d=True
                                     )
            backbone_model.build_model()

    elif FLAGS.backbone_model == 'c2d':
        # Resnet-50 C2D model
        with tf.variable_scope(("resnet_c2d")):
            backbone_model = NLModel_2D(FLAGS, video_pl_all_tower, FLAGS.num_classes,
                                        target_pl=target_pl_all_tower,
                                        is_training=is_training_pl,
                                        num_towers=num_towers,
                                        tower_index=tower_i,
                                        graph_models=graph_models,
                                        i3d=True
                                        )
            backbone_model.build_model()

    elif FLAGS.backbone_model == 'non-local':
        # Resnet-50 Non-Local model
        with tf.variable_scope(("resnet_i3d")):
            backbone_model = NLModel(FLAGS, video_pl_all_tower,
                                     FLAGS.num_classes,
                                     target_pl=target_pl_all_tower,
                                     is_training=is_training_pl,
                                     num_towers=num_towers,
                                     tower_index=tower_i,
                                     graph_models=graph_models,
                                     i3d=False
                                     )
            backbone_model.build_model()

    elif FLAGS.backbone_model == 'simple_c2d':
        # Simple Convolutional Model for MNIST dataset
        backbone_model = SimpleConvModel(FLAGS, video_pl_all_tower,
                                         FLAGS.num_filters, FLAGS.num_classes,
                                         target_pl=target_pl_all_tower,
                                         is_training=is_training_pl,
                                         num_towers=num_towers,
                                         tower_index=tower_i,
                                         graph_models=graph_models
                                         )
        backbone_model.build_model()
    return backbone_model, graph_models


def run():
    # prepare tensorflow
    global_step = tf.Variable(0, trainable=False, name="global_step")
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    num_gpus = len(gpus)

    if num_gpus > 0:
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        num_towers = 1
        device_string = '/cpu:%d'

    gc = PatchConstructor(FLAGS)
    adj_matrix, _ = gc.get_adjacency()

    # create datasets
    if FLAGS.dataset == 'smt-smt':
        dc = SmtDatasetCreator(num_towers, FLAGS.batch_size)

        # test set
        big_test_images = sorted(glob(FLAGS.test_dataset + "/*batch_0.jpeg"))
        len_big_test_images = len(big_test_images)
        # subset of len_eval_test videos for more frequently evaluation
        test_images = big_test_images[:FLAGS.len_eval_test]

        train_images = glob(FLAGS.train_dataset + "/*.jpeg")
        random.shuffle(train_images)
        train_eval_images = train_images.copy()

        train_iterator, train_handle = dc.create_dataset(
            train_images, FLAGS.gt_train_file,
            num_towers, sess, name="train", training=True)
    else:
        dc = MnistDatasetCreator(num_towers, FLAGS.batch_size)

        no_videos_per_pickle = 1000
        train_files = glob(FLAGS.train_dataset + '/*pickle')
        crt_train_videos = tf.placeholder(
            tf.int32,
            [no_videos_per_pickle, FLAGS.video_num_frames,
             FLAGS.dim_h, FLAGS.dim_w, 1])
        crt_train_labels = tf.placeholder(
            tf.float64,
            [no_videos_per_pickle, 1, FLAGS.num_classes])

        # this dataset does not contains video ids.
        # create dummy ids for compatibility with Smt-Smt code
        video_ids = np.zeros(crt_train_labels.shape)

        train_iterator, train_handle = dc.build_train_dataset(
            crt_train_videos, crt_train_labels, video_ids,
            num_towers, sess)

        train_eval_images = [train_files[0]]

        # test set
        big_test_images = test_images = glob(
            FLAGS.test_dataset + '/*pickle')[:5]
        len_big_test_images = len(big_test_images) * no_videos_per_pickle

    test_iterator, test_handle = dc.create_dataset(
        test_images, FLAGS.gt_test_file, num_towers,
        sess, len_eval=FLAGS.len_eval_test,
        name="test_eval")

    big_test_iterator, big_test_handle = dc.create_dataset(
        big_test_images, FLAGS.gt_test_file, num_towers,
        sess, name="big_test_eval")

    train_eval_iterator, train_eval_handle = dc.create_dataset(
        train_eval_images, FLAGS.gt_train_file, num_towers,
        sess, name="train_eval")

    handle = tf.placeholder(tf.string, shape=[])
    # we only need the shape and type of dataset_test
    iterator = tf.data.Iterator.from_string_handle(
        handle, dc.get_type(), dc.get_shape())
    video_pl_merged_all, target_pl_all, video_ids = iterator.get_next()

    video_pl_all = dc.preprocessing(video_pl_merged_all)
    tower_gradients, tower_accuracy, tower_losses, tower_accuracy_top_k = [], [], [], []
    adj_matrix_pl = tf.placeholder(
        tf.float32,
        [FLAGS.batch_size * FLAGS.num_eval_clips, gc.num_nodes+1, gc.num_nodes+1])
    is_training_pl = tf.placeholder(tf.bool, ())

    # BUILD model
    for tower_i in range(num_towers):
        video_pl_all_tower = video_pl_all
        target_pl_all_tower = target_pl_all
        with tf.device(device_string % tower_i):
            with (tf.variable_scope(("tower"), reuse=True if tower_i > 0 else None)):
                with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus != 1 else "/gpu:0")):
                    backbone_model, graph_models = create_models(
                        num_towers, tower_i, video_pl_all_tower, target_pl_all_tower, adj_matrix_pl, is_training_pl, gc)
                    all_graphs = [value for key,
                                  value in graph_models.items() if value]
                    # select one graph model just to access shared placeholders
                    last_graph_model = all_graphs[-1] if len(
                        all_graphs) > 0 else None

                    current_loss, current_accuracy, current_accuracy_top_k = compute_loss(
                        backbone_model)

                    tower_losses.append(current_loss)
                    tower_accuracy.append(current_accuracy)
                    tower_accuracy_top_k.append(current_accuracy_top_k)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    if update_ops:
                        with tf.control_dependencies(update_ops):
                            barrier = tf.no_op(name="gradient_barrier")
                            with tf.control_dependencies([barrier]):
                                current_loss = tf.identity(current_loss)

                    trainable_vars = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES)
                    trainable_vars = [
                        var for var in trainable_vars if 'resize_filter' not in var.name]
                    print('Trainable variables')
                    [print(v) for v in trainable_vars]

                    lr_placeholder = tf.placeholder(tf.float32, ())
                    # create train operation
                    optimizer = tf.train.MomentumOptimizer(
                        learning_rate=lr_placeholder,
                        use_nesterov=True,
                        momentum=0.9)
                    gradients = optimizer.compute_gradients(current_loss,
                                                            colocate_gradients_with_ops=False,
                                                            var_list=trainable_vars)

                    tower_gradients.append(gradients)

    towers_mean_loss = tf.reduce_mean(tf.stack(tower_losses))
    towers_mean_accuracy = tf.reduce_mean(tf.stack(tower_accuracy))
    merged_gradients = combine_gradients(tower_gradients)

    # add summaries
    tf.summary.scalar('loss', towers_mean_loss)
    tf.summary.scalar('accuracy', towers_mean_accuracy)
    eval_mean_acc_summary = tf.Summary()
    eval_mean_acc_summary.value.add(
        tag='eval_mean_accuracy', simple_value=None)
    epoch_summary = tf.Summary()
    epoch_summary.value.add(tag='epoch', simple_value=None)
    train_mean_acc_summary = tf.Summary()
    train_mean_acc_summary.value.add(
        tag='train_mean_accuracy', simple_value=None)

    train_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
    valid_writer = tf.summary.FileWriter(
        FLAGS.model_dir + "/valid/", sess.graph)
    valid_writer_all = tf.summary.FileWriter(
        FLAGS.model_dir + "/valid_all/", sess.graph)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(
            merged_gradients, global_step=global_step)

    all_summaries = tf.get_collection(key=tf.GraphKeys.SUMMARIES)
    merged_summaries = tf.summary.merge(inputs=all_summaries)

    saver = tf.train.Saver(max_to_keep=None)
    initialize(sess, latest_checkpoint)

    best_acc = 0
    global_step_val = 0

    if FLAGS.mode == 'train':
        for epoch in range(FLAGS.num_epochs):
            print('_' * 120)
            print(f'Epoch: {epoch}')
            print('_' * 120)

            epoch_summary.value[0].simple_value = epoch
            train_writer.add_summary(epoch_summary, global_step_val)

            # For MNIST we have to manually iterate through the pickle files.
            #   each pickle contains several videos
            #   we should read each pickle independently and feed into the dataset initializer
            # For smt-smt the iterator takes care of all the videos automatically
            if FLAGS.dataset == 'smt-smt':
                num_files_per_epoch = 1         # tf.dataset cope with epochs and stuff
                sess.run(train_iterator.initializer)
            elif FLAGS.dataset == 'mnist':
                num_files_per_epoch = len(train_files)
                random.shuffle(train_files)

            for file_no in range(num_files_per_epoch):
                if FLAGS.dataset == 'mnist':
                    train_videos, train_labels = utils.read_data_mnist(
                        train_files[file_no], num_classes=FLAGS.num_classes)
                    sess.run(train_iterator.initializer, feed_dict={
                             crt_train_videos: train_videos, crt_train_labels: train_labels})
                while(True):
                    try:
                        feed_dict = {
                            backbone_model.is_training: True,
                            handle: train_handle,
                            lr_placeholder: FLAGS.learning_rate
                        }
                        # feed placeholder for all the graphs (they are shared between all of them)
                        if last_graph_model:
                            feed_dict.update({
                                last_graph_model.adj_matrix_pl: adj_matrix,
                                last_graph_model.is_training: True
                            })

                        # TRAIN STEP
                        [_, summaries_val, loss_val,
                         global_step_val, acc_val] = sess.run([
                             train_op, merged_summaries,
                             towers_mean_loss, global_step,
                             towers_mean_accuracy,
                         ], feed_dict=feed_dict)


                        if (global_step_val % FLAGS.print_every == 0):
                            print('Step ' + str(global_step_val) +
                                  ' loss is : ', loss_val, 'acc train: ', acc_val)
                            sys.stdout.flush()

                        if global_step_val % FLAGS.save_every == 0:
                            save_model(saver, FLAGS.model_dir,
                                       sess, global_step_val)

                        if (global_step_val % FLAGS.eval_every == 0):
                            # EVALUATE on len_eval_train of the train set
                            train_writer.add_summary(
                                summaries_val, global_step_val)
                            sess.run(train_eval_iterator.initializer)
                            _ = eval_model(
                                sess, backbone_model, last_graph_model, merged_summaries,
                                train_writer, global_step_val, eval_mean_acc_summary, adj_matrix,
                                num_towers=num_towers,
                                tower_accuracy=tower_accuracy,
                                tower_accuracy_top_k=tower_accuracy_top_k,
                                tower_loss=tower_losses,
                                handle_pl=handle, handle_val=train_eval_handle,
                                len_test=FLAGS.len_eval_train,
                                name='train')

                            # EVALUATE on small validation set
                            sess.run(test_iterator.initializer)
                            eval_acc = eval_model(
                                sess, backbone_model, last_graph_model,
                                merged_summaries, valid_writer, global_step_val,
                                eval_mean_acc_summary, adj_matrix,
                                num_towers=num_towers,
                                tower_accuracy=tower_accuracy,
                                tower_accuracy_top_k=tower_accuracy_top_k,
                                tower_loss=tower_losses,
                                handle_pl=handle, handle_val=test_handle,
                                len_test=FLAGS.len_eval_test,
                                name='valid')

                            # EVALUATE on the entire validation set
                            if (global_step_val % (10 * FLAGS.eval_every) == 0):
                                sess.run(big_test_iterator.initializer)
                                eval_acc = eval_model(
                                    sess, backbone_model, last_graph_model, merged_summaries,
                                    valid_writer_all, global_step_val, eval_mean_acc_summary,
                                    adj_matrix,
                                    num_towers=num_towers,
                                    tower_accuracy=tower_accuracy,
                                    tower_accuracy_top_k=tower_accuracy_top_k,
                                    tower_loss=tower_losses,
                                    handle_pl=handle, handle_val=big_test_handle,
                                    len_test=len_big_test_images,
                                    name='all_valid')

                                if eval_acc >= best_acc:
                                    best_acc = eval_acc
                                    save_model(saver, FLAGS.model_dir,
                                               sess, global_step_val)

                    except Exception as e:
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(
                            exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        break
    else:
        # EVALUATE on the entire validation set
        sess.run(big_test_iterator.initializer)
        eval_acc = eval_model(
            sess, backbone_model, last_graph_model, merged_summaries,
            valid_writer_all, global_step_val, eval_mean_acc_summary, adj_matrix,
            num_towers=num_towers,
            tower_accuracy=tower_accuracy,
            tower_accuracy_top_k=tower_accuracy_top_k,
            tower_loss=tower_losses,
            handle_pl=handle, handle_val=big_test_handle,
            len_test=len_big_test_images,
            name='final_all_valid')


run()
