import tensorflow as tf
from tensorflow import flags
import time
import os
import pickle
import random
import pdb
import pylab
import numpy as np
# import scipy.misc
import scipy
from os.path import exists
from os import mkdir, system
from loader import MNIST, load_mnist

FLAGS = flags.FLAGS

flags.DEFINE_integer("max_digits_in_frame", 5, "number of digits in a frame")
flags.DEFINE_integer("frame_size", 100, "size of the considered image")
flags.DEFINE_integer("dataset_multiplier", 10,
                     "make dataset X times larger than MNIST")
flags.DEFINE_integer(
    "save_first_n_videos",
    20,
    "save as .png the individual frames of the first X videos",
)
flags.DEFINE_integer(
    "sync_dist", -1, "maximum distance between the syncronous digits"
)  # 20
flags.DEFINE_integer("noise_parts_size", 8,
                     "size of the digits parts used as noise")
flags.DEFINE_integer("no_noise_parts", 0, "number of noise parts")
flags.DEFINE_integer("video_frames", 10, "number of frames in a video")

flags.DEFINE_string("split", "train", "train or test")
flags.DEFINE_string("dataset_name", "digits_sync", "name of the dataset")
flags.DEFINE_string("dataset_path", "./datasets/", "path to save the dataset")
flags.DEFINE_boolean("predefined", True,
                     "create random paths (moves) or load from predefined ones")
flags.DEFINE_string("predefined_path", "./create_datasets/predefined_path_train.pickle",
                    "file for predefined digits paths (moves)")


# create the set of labels
# a label is defined by a set of two digits
# final label is for the case with all digits moving randomply
# 10 * 9 / 2 + 1 = 45 + 1 labels

def get_labels_dict_sincron():
    # create labels
    label = 0
    labels_dict = {}
    for i in range(10):
        for j in range(i):
            labels_dict[(i, j)] = label
            labels_dict[(j, i)] = label
            label += 1
    labels_dict[(-1, -1)] = label
    label += 1
    return labels_dict, label


def get_labels_sincron(labels_dict, sync_digits, rr=None):
    # based on the digits get the label of the video
    sync_digits = tuple(sync_digits)
    current_label = labels_dict[sync_digits]
    # with chance 1 / number_of_labels make move the digits randomply
    if rr == None:
        rr = np.random.rand()
    if rr <= 1.0 / (len(labels_dict) + 1):
        sincron = "random"
        current_label = labels_dict[(-1, -1)]  # len(labels_dict)
    else:
        sincron = "sincron"
    return current_label, sincron


def move_random_ind(coords, sz, yy, move_range=10):
    for ind in range(coords.shape[0]):
        dtop = np.clip(0 - coords[ind][0], -move_range, move_range)
        dbot = np.clip((sz - yy - 1) - coords[ind][0], -move_range, move_range)

        dleft = np.clip(0 - coords[ind][1], -move_range, move_range)
        dright = np.clip(
            (sz - yy - 1) - coords[ind][1], -move_range, move_range)

        coords[ind][0] = np.clip(
            coords[ind][0] +
            np.random.randint(dtop, dbot, size=[1]), 0, sz - yy - 1
        )
        coords[ind][1] = np.clip(
            coords[ind][1] +
            np.random.randint(dleft, dright, size=[1]), 0, sz - yy - 1
        )
    return coords


def move_random_sincr(coords, sz, yy, move_range=10):
    if coords.shape[0] != 2:
        print("Should be two digits for sinc")
        return 0
    dtop = -move_range
    dbot = move_range
    dleft = -move_range
    dright = move_range

    for ind in range(2):
        dtop = max(dtop, np.clip(0 - coords[ind][0], -move_range, move_range))
        dbot = min(
            dbot, np.clip((sz - yy - 1) -
                          coords[ind][0], -move_range, move_range)
        )

        dleft = max(dleft, np.clip(
            0 - coords[ind][1], -move_range, move_range))
        dright = min(
            dright, np.clip(
                (sz - yy - 1) - coords[ind][1], -move_range, move_range)
        )
    if dleft < dright:
        coords[:2, 1] = np.clip(
            coords[:2, 1] +
            np.random.randint(dleft, dright, size=[1]), 0, sz - yy - 1
        )
    if dtop < dbot:
        coords[:2, 0] = np.clip(
            coords[:2, 0] +
            np.random.randint(dtop, dbot, size=[1]), 0, sz - yy - 1
        )
    return coords


def make_video_sincron(data):
    batch_ind = 0
    batch_no = 0

    multiplier = FLAGS.dataset_multiplier
    mnist_dim = data[FLAGS.split + "_imgs"].shape[0]
    if not FLAGS.predefined:
        random_paths = {}

        random_paths['digit_position'] = np.zeros(
            (multiplier, mnist_dim)).astype(np.uint8)
        random_paths['coords_digits'] = np.zeros(
            (multiplier, mnist_dim, FLAGS.video_frames, FLAGS.max_digits_in_frame, 2)).astype(np.uint8)
        random_paths['noise_coords'] = np.zeros(
            (multiplier, mnist_dim, FLAGS.video_frames, FLAGS.no_noise_parts, 2)).astype(np.uint8)
        random_paths['ind_rand_part'] = np.zeros(
            (multiplier, mnist_dim, FLAGS.no_noise_parts)).astype(np.uint32)
        random_paths['ind_digits'] = np.zeros(
            (multiplier, mnist_dim, FLAGS.max_digits_in_frame)).astype(np.uint32)
        random_paths['set_digits'] = np.zeros(
            (multiplier, mnist_dim, FLAGS.max_digits_in_frame - 1)).astype(np.uint8)
        random_paths['sincron_prob'] = np.zeros(
            (multiplier, mnist_dim)).astype(np.uint8)

    else:
        savepath = FLAGS.predefined_path
        with open(savepath, 'rb') as f:
            random_paths = pickle.load(f)

    labels_dict, num_classes = get_labels_dict_sincron()
    dataset_path = (f"{FLAGS.dataset_path}/{FLAGS.dataset_name}_split_{FLAGS.split}_sync_dist_{FLAGS.sync_dist}_"
                    f"num_classes_{num_classes}_"
                    f"no_digits_{FLAGS.max_digits_in_frame}_"
                    f"no_noise_parts_{FLAGS.no_noise_parts}/")
    print(dataset_path)
    print(f"Number of classes:{num_classes}")

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        os.mkdir(dataset_path + "/images/")

    # create a batch of videos having a fixed size
    # each video has one global label
    videos = np.zeros((1000, FLAGS.video_frames, 64, 64)).astype(np.int32)
    labels = np.zeros((1000, 1))

    # digits types (0 - 9) and their position
    videos_digits = np.zeros((1000, FLAGS.max_digits_in_frame))
    videos_digits_coords = np.zeros(
        (1000, FLAGS.video_frames, FLAGS.max_digits_in_frame, 2))

    # labels_sample_idx keps indexes of all images of a certain type
    labels_sample_idx = {}
    for ind, l in enumerate(data[FLAGS.split + "_labels"]):
        if l in labels_sample_idx:
            labels_sample_idx[l].append(ind)
        else:
            labels_sample_idx[l] = [ind]

    for rep in range(FLAGS.dataset_multiplier):
        for ind in range(data[FLAGS.split + "_imgs"].shape[0]):
            if rep == 0 and ind <= FLAGS.save_first_n_videos:
                print(f'video: {ind} saving all frames as .png')
            if rep == 0 and ind == FLAGS.save_first_n_videos:
                print(f'save videos in .pickle files')

            current_digit_class = data[FLAGS.split + "_labels"][ind]

            # current video
            video = np.zeros((FLAGS.video_frames, 64, 64)).astype(np.int32)
            if FLAGS.predefined:
                digit_position = random_paths['digit_position'][rep, ind]
                ind_digits = random_paths['ind_digits'][rep, ind]
                set_digits = random_paths['set_digits'][rep, ind]
                ind_rand_part = random_paths['ind_rand_part'][rep, ind]
                digits_in_video = [current_digit_class] + list(set_digits)

            else:
                # put digits in front / back of the noise
                #digit_position = random.choice(["front", "back"])
                digit_position = random.choice([1, 0])  # front / back

                if FLAGS.no_noise_parts == 0:
                    digit_position = 1

                # position coordinates of all the digits in the frame
                coords_digits = np.random.randint(
                    FLAGS.frame_size - 28, size=[FLAGS.max_digits_in_frame, 2]
                )
                # coordinates of the noise (digits parts)
                noise_coords = np.random.randint(
                    FLAGS.frame_size - FLAGS.noise_parts_size,
                    size=[FLAGS.no_noise_parts, 2],
                )

                # get indices for the parts / digits used in current video
                ind_rand_part = np.random.randint(
                    data[FLAGS.split + "_imgs"].shape[0], size=[FLAGS.no_noise_parts]
                )

                # select digits used in the current video
                # use a set of different digits
                set_digits = np.array(
                    list(set(range(10)) - set([current_digit_class])))
                perm = np.random.permutation(9)
                set_digits = set_digits[perm[: FLAGS.max_digits_in_frame - 1]]

                # class of the digits used
                digits_in_video = [current_digit_class] + list(set_digits)
                # select from MNIST digits with the selected classes
                ind_digits = [ind]
                # select random images from all the digits images of the selected classes
                for digit_cl in set_digits:
                    all_cl_inx = labels_sample_idx[digit_cl]
                    idx = np.random.randint(len(all_cl_inx))
                    ind_digits.append(all_cl_inx[idx])

                random_paths['digit_position'][rep, ind] = digit_position

                random_paths['ind_digits'][rep, ind] = ind_digits
                random_paths['set_digits'][rep, ind] = set_digits

                random_paths['coords_digits'][rep, ind] = coords_digits
                random_paths['noise_coords'][rep, ind] = noise_coords
                random_paths['ind_rand_part'][rep, ind] = ind_rand_part

            sync_digits = [current_digit_class] + list(set_digits[:1])
            current_label, sincron = get_labels_sincron(
                labels_dict, sync_digits)

            if FLAGS.predefined:
                current_label, sincron = get_labels_sincron(labels_dict, sync_digits,
                                                            random_paths['sincron_prob'][rep, ind])
            else:
                random_paths['sincron_prob'][rep,
                                             ind] = 1 if sincron == 'sincron' else 0

            # if distance between the sync digits is not random
            # set the distance between them
            # if sincron == "sincron" and FLAGS.sync_dist > 0:
            #     tmp_coords_digits = move_sync_digits_closer(
            #         coords_digits[:2],
            #         FLAGS.frame_size,
            #         yy=28,
            #         move_range=FLAGS.sync_dist,
            #     )
            #     if np.any(tmp_coords_digits == None):
            #         continue
            #     coords_digits[:2] = tmp_coords_digits

            videos_digits[batch_ind] = np.array(digits_in_video)

            for frame_no in range(FLAGS.video_frames):
                # current frame of the video
                frame = np.zeros((FLAGS.frame_size, FLAGS.frame_size))
                if not FLAGS.predefined:
                    noise_coords = move_random_ind(
                        noise_coords,
                        FLAGS.frame_size,
                        FLAGS.noise_parts_size,
                        move_range=10,
                    )

                    if sincron == "sincron":
                        coords_digits[:2] = move_random_sincr(
                            coords_digits[:2], FLAGS.frame_size, yy=28
                        )
                        coords_digits[2:] = move_random_ind(
                            coords_digits[2:], FLAGS.frame_size, yy=28, move_range=10
                        )
                    else:
                        coords_digits = move_random_ind(
                            coords_digits, FLAGS.frame_size, yy=28, move_range=10
                        )

                    random_paths['coords_digits'][rep,
                                                  ind, frame_no] = coords_digits
                    random_paths['noise_coords'][rep,
                                                 ind, frame_no] = noise_coords
                else:
                    coords_digits = random_paths['coords_digits'][rep,
                                                                  ind, frame_no]
                    noise_coords = random_paths['noise_coords'][rep,
                                                                ind, frame_no]
                videos_digits_coords[batch_ind, frame_no] = coords_digits

                # randomly move the noise
                for i in range(FLAGS.no_noise_parts):
                    patch = frame[
                        noise_coords[i, 0]: noise_coords[i, 0]
                        + FLAGS.noise_parts_size,
                        noise_coords[i, 1]: noise_coords[i, 1] + FLAGS.noise_parts_size,
                    ]
                    d1 = patch.shape[0]
                    d2 = patch.shape[1]
                    crop = data[FLAGS.split + "_imgs"][ind_rand_part[i]][
                        14 - d1 // 2: 14 + d1 - d1 // 2,
                        14 - d2 // 2: 14 + d2 - d2 // 2,
                    ]
                    frame[
                        noise_coords[i, 0]: noise_coords[i, 0]
                        + FLAGS.noise_parts_size,
                        noise_coords[i, 1]: noise_coords[i, 1] + FLAGS.noise_parts_size,
                    ] = crop
                # move the digits
                for i_d, ind_digit in enumerate(ind_digits):
                    if ind_digit == None:
                        continue
                    # select digit image
                    img = data[FLAGS.split + "_imgs"][ind_digit]
                    digit_c = coords_digits[i_d]
                    # select patch of the frame
                    frame_patch = frame[
                        digit_c[0]: digit_c[0] + 28, digit_c[1]: digit_c[1] + 28
                    ]
                    if digit_position == 0:  # back
                        mask = frame_patch > 0
                    else:
                        mask = img < 10
                    # dist = scipy.ndimage.distance_transform_cdt(mask, metric='taxicab')
                    # dist = dist / dist.max()
                    # dist = 1.0 - (dist == 0.0).astype(np.float32)
                    dist = mask.astype(float)
                    frame[
                        digit_c[0]: digit_c[0] + 28, digit_c[1]: digit_c[1] + 28
                    ] = (dist * frame_patch + (1 - dist) * img)

                frame = np.clip(frame, 0, 255)
                # save as png the frames of the first videos
                if rep == 0 and ind <= FLAGS.save_first_n_videos:
                    pylab.imshow(
                        scipy.misc.imresize(frame, size=(64, 64)), cmap="Greys_r"
                    )
                    savepath = (f"/images/video{ind}_{frame_no}_"
                                f"{sincron}_sync_digits_{sync_digits[0]}_"
                                f"{sync_digits[1]}_all_digits:{digits_in_video}.png")
                    pylab.savefig(
                        dataset_path
                        + savepath
                    )
                video[frame_no] = scipy.misc.imresize(frame, size=(64, 64)).astype(
                    np.int32
                )

            videos[batch_ind] = video
            labels[batch_ind] = current_label
            batch_ind += 1
            # save video batches in pickle
            if ind > 0 and (ind + 1) % 1000 == 0:
                savepath = FLAGS.predefined_path
                with open(savepath, 'wb') as f:
                    pickle.dump(random_paths, f)

                print(f'Saved {ind} videos')
                with open(dataset_path + f"/data_{batch_no}.pickle", "wb") as f:
                    save_dict = {
                        "videos": videos.astype(np.int32),
                        "labels": labels,
                        "videos_digits": videos_digits.astype(np.int32),
                        "videos_digits_coords": videos_digits_coords,
                    }
                    pickle.dump(save_dict, f)

                videos = np.zeros((1000, FLAGS.video_frames, 64, 64))
                videos_digits = np.zeros((1000, FLAGS.max_digits_in_frame))
                videos_digits_coords = np.zeros(
                    (1000, FLAGS.video_frames, FLAGS.max_digits_in_frame, 2)
                )

                batch_ind = 0
                batch_no += 1


if __name__ == "__main__":
    data = load_mnist()

    if FLAGS.split == "test":
        FLAGS.dataset_multiplier = 1

    make_video_sincron(data)
