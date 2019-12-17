# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:35:18 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import os


def load_and_preprocess_image(path, shape, to_gray=False):
    image_string = tf.io.read_file(path)
    image = tf.cond(tf.image.is_jpeg(image_string),
                    lambda: tf.image.decode_jpeg(image_string, channels=3),
                    lambda: tf.image.decode_png(image_string, channels=3))
    image = tf.image.resize(image, shape)
    image = tf.image.rot90(image)
    image /= 255.0

    if to_gray:
        image = tf.image.rgb_to_grayscale(image)

    return image

def augmentation(img, label):
    img_aug = tf.image.random_flip_left_right(img)
    img_aug = tf.image.random_flip_up_down(img_aug)

    return img_aug, label

def Get_TF_Dataset(BATCH_SIZE):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Get all data path
    img_path_list = []
    label_list = []
    with open('train.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'ID':
                continue
            img_path_list.append('./train_images/' + row[0])
            label_list.append(int(row[1]))

    # You might want to manually train/val split here
    img_path_train = img_path_list[:8000]
    label_train = label_list[:8000]
    img_path_val = img_path_list[8000:]
    label_val = label_list[8000:]

    # Make training tf.dataset
    img_path_train_ds = tf.data.Dataset.from_tensor_slices(img_path_train) # path dataset (strings)
    img_train_ds = img_path_train_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256]), num_parallel_calls=AUTOTUNE) # image dataset
    label_train_ds = tf.data.Dataset.from_tensor_slices(label_train) # CAST ????

    train_ds = tf.data.Dataset.zip((image_train_ds, label_train_ds))
    train_ds = train_ds.map(augmentation, AUTOTUNE) # augment only for training dataset

    # Make validation tf.dataset
    img_path_val_ds = tf.data.Dataset.from_tensor_slices(img_path_val) # path dataset (strings)
    img_val_ds = img_path_val_ds.map(lambda x: load_and_preprocess_image(x, shape=[256, 256]), num_parallel_calls=AUTOTUNE) # image dataset
    label_val_ds = tf.data.Dataset.from_tensor_slices(label_val) # CAST ????

    val_ds = tf.data.Dataset.zip((image_val_ds, label_val_ds))

    # Settings for the dataset
    train_ds = train_ds.shuffle(buffer_size=2048)