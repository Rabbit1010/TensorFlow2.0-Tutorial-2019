# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:11:52 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


# Get path to all files
img_path_list = []
label_list = []

for fname in os.listdir('./data/train/'):
    if fname.startswith('cat'):
        img_path_list.append('./data/train/'+fname)
        label_list.append(0) # simultaneously get the label from filename
    elif fname.startswith('dog'):
        img_path_list.append('./data/train/'+fname)
        label_list.append(1)

    if len(label_list)>5: break # for simplicity

print('Training samples:', len(label_list))
print('Cat samples:', label_list.count(0))
print('Dog samples:', label_list.count(1))

# Construct tf.data.Dataset
path_ds = tf.data.Dataset.from_tensor_slices((img_path_list, label_list))

# The function to parse image data
def load_and_preprocess_img(path, label, shape):
    image_string = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, shape)
    image = image / 255.0

    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE # let TensorFlow tune the degree of parallel threads
# map the function to the dataset (with parallellism)
image_label_ds = path_ds.map(lambda x,y: load_and_preprocess_img(x, y, [256,256]), num_parallel_calls=AUTOTUNE)

# iterate through the dataset
for image, label in image_label_ds:
    plt.figure()
    plt.imshow(image.numpy())
    plt.title(label.numpy())