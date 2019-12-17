# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:11:52 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


AUTOTUNE = tf.data.experimental.AUTOTUNE

# Construct tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7,8,9,10])

# Shuffle the samples
train_ds = train_ds.shuffle(buffer_size=10)

# Batch the samples
train_ds = train_ds.batch(3)

# Let the dataset fetch in the background while the model is training
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

for sample in train_ds:
    print(sample.numpy())