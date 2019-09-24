import numpy as np
from array import array

import os
import struct
from os.path import exists
from os import mkdir, system


class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render


MNIST_PATH = "./MNIST"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
BASE_URL = "http://yann.lecun.com/exdb/mnist"


def download_mnist():
    if not exists(MNIST_PATH):
        mkdir(MNIST_PATH)
    for f in FILES:
        if not exists("%s/%s" % (MNIST_PATH, f[:-3])):
            system("wget %s/%s -O %s/%s" % (BASE_URL, f, MNIST_PATH, f))
            system("gunzip %s/%s" % (MNIST_PATH, f[:-3]))


def preprocess(train_imgs, test_imgs):
    avg = np.mean(train_imgs)
    dev = np.std(train_imgs)

    train_imgs -= avg
    train_imgs /= dev
    test_imgs -= avg
    test_imgs /= dev


def load_mnist():
    download_mnist()
    mnist_data = MNIST(MNIST_PATH)
    train_imgs, train_labels = mnist_data.load_training()
    test_imgs, test_labels = mnist_data.load_testing()
    data = {}
    data["train_imgs"] = np.array(train_imgs, dtype="f").reshape(60000, 28, 28)
    data["test_imgs"] = np.array(test_imgs, dtype="f").reshape(10000, 28, 28)
    data["train_labels"] = np.array(train_labels)
    data["test_labels"] = np.array(test_labels)

    data["train_no"] = 60000
    data["test_no"] = 10000

    return data
