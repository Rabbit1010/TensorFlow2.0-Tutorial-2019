#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:07:47 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def extract_fn(data_record):
    features_description = {
        "image/original" :tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(data_record, features_description)

    image = tf.io.decode_jpeg(sample['image/original'], channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image/255.0

    #Augmentation
    image = tf.image.random_flip_left_right(image)
    image = image[:-112, 56:-56]

    return image

def Get_DS(BATCH_SIZE=16):
    # Get all path to tf records
    tfrecord_train_list = []
    tfrecord_val_list = []
    for fname in os.listdir('./data/'):
        if fname.lower().startswith('train'):
            tfrecord_train_list.append('./data/' + fname)
        elif fname.lower().startswith('val'):
            tfrecord_val_list.append('./data/' + fname)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset_train = tf.data.TFRecordDataset(tfrecord_train_list)
    dataset_train = dataset_train.map(extract_fn)
    dataset_train = dataset_train.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    dataset_val = tf.data.TFRecordDataset(tfrecord_val_list)
    dataset_val = dataset_val.map(extract_fn)
    dataset_val = dataset_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset_train, dataset_val

if __name__ == '__main__':
    dataset_train, dataset_val = Get_DS()

    for batch in dataset_train.take(1):
        image = batch.numpy()

    i_pic = 0
    plt.imshow(np.squeeze(image[i_pic,:,:,:]))
    plt.show()

    for batch in dataset_val.take(1):
        image = batch.numpy()

    i_pic = 0
    plt.imshow(np.squeeze(image[i_pic,:,:,:]))
    plt.show()