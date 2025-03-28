# Code modified from https://github.com/foamliu/Colorful-Image-Colorization/blob/master/data_generator.py

import os
import random
from random import shuffle

import cv2
import numpy as np
import sklearn.neighbors as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence

from config import batch_size, img_rows, img_cols, nb_neighbors

image_X_folder = '/src/data/train_X/' # Folder of input training data
image_y_folder = '/src/data/train_y/' # Folder of ground truth data to evaluate outputs against


def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    ab = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    # Smooth the weights with a gaussian kernel
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    # format the tar get
    y = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts
    y = y.reshape(h, w, nb_q)
    return y


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        if usage == 'train':
            names_file = '/src/data/train_names.txt'
        else:
            names_file = '/src/data/valid_names.txt'

        with open(names_file, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)

        # Load the array of quantized ab value
        q_ab = np.load('/src/zhang/resources/pts_in_hull.npy')
        self.nb_q = q_ab.shape[0]
        # Fit a NN to q_ab
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        out_img_rows, out_img_cols = img_rows // 4, img_cols // 4

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, out_img_rows, out_img_cols, self.nb_q), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            x = read_lab(os.path.join(image_X_folder, name))
            y = read_lab(os.path.join(image_y_folder, name))

            y = cv2.resize(y, (out_img_cols, out_img_rows), cv2.INTER_CUBIC)
            # Before: 42 <=a<= 226, 20 <=b<= 223
            # After: -86 <=a<= 98, -108 <=b<= 95
            y_ab = y[:, :, 1:].astype(np.int32) - 128

            y = get_soft_encoding(y_ab, self.nn_finder, self.nb_q)

            if np.random.random_sample() > 0.5:
                x = np.fliplr(x)
                y = np.fliplr(y)

            batch_x[i_batch] = x
            batch_y[i_batch] = y

            i += 1

        print(str(batch_x.shape) + ' ' + str(batch_y.shape))

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

def read_lab(filename):
    # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
    bgr = cv2.imread(filename)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab

def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')


def split_data():
    names = [f for f in os.listdir(image_X_folder) if f.lower().endswith('.png')]

    # num_samples = len(names)  # 1341430
    num_samples = 1000
    print('num_samples: ' + str(num_samples))

    num_train_samples = int(num_samples * 0.992)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    # with open('names.txt', 'w') as file:
    #     file.write('\n'.join(names))

    with open('/src/data/valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('/src/data/train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    split_data()