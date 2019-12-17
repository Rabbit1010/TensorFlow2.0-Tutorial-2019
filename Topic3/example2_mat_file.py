#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:13:13 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa


def Get_Dataset():
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def load_mat_file(path):
        # img = scipy.io.loadmat(path)['img']
        # img = img / 255.0
        
        # img_BW = scipy.io.loadmat(path)['img_BW']
        # img_BW = img_BW / 255.0
        
        def load_mat(path):
            img = scipy.io.loadmat(path.numpy())['img']
            img = img / 255.0
            
            return img

        def load_mat_BW(path):
            img = scipy.io.loadmat(path.numpy())['img_BW']
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)
            
            return img
        
        img = tf.py_function(load_mat, [path], (tf.float32))
        img_BW = tf.py_function(load_mat_BW, [path], (tf.float32))
        
        return img, img_BW
    
    def augmentation(img, img_BW):
        def sequential_aug(img1, img2):
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ], random_order=False)
            
            seq = seq.to_deterministic() # freeze random state
            
            img1 = seq.augment_image(img1)
            img2 = seq.augment_image(img2)
            
            return img1, img2
        
        img, img_BW = tf.py_function(sequential_aug, [img, img_BW], (tf.float32, tf.float32))
        
        return img, img_BW
    
    file_list = ['./a.mat', './b.mat']
    path_ds = tf.data.Dataset.from_tensor_slices(file_list)
    paired_ds = path_ds.map(load_mat_file, AUTOTUNE)
    paired_ds = paired_ds.cache() # since loading .mat file from Python is a heavy operation
    paired_ds = paired_ds.map(augmentation, AUTOTUNE)
    
    # Add all settings
    # paired_ds = paired_ds.shuffle(2)
    # paired_ds = paired_ds.batch(1)
    paired_ds = paired_ds.prefetch(AUTOTUNE)
    
    return paired_ds

# Unit testing for data generator
if __name__ == '__main__':
    ds = Get_Dataset()
    
    for img, img_BW in ds:
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(img_BW[:,:,0], cmap='gray')