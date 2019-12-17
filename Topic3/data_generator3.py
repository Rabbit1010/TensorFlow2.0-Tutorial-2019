# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:11:52 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import imgaug.augmenters as iaa


# The function to parse image data
def load_and_preprocess_img(path, label, shape):
    image_string = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, shape)
    image = image / 255.0

    return image, label

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

    if len(label_list)>3: break # for simplicity

print('Training samples:', len(label_list))
print('Cat samples:', label_list.count(0))
print('Dog samples:', label_list.count(1))

# Construct tf.data.Dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
path_ds = tf.data.Dataset.from_tensor_slices((img_path_list, label_list))
image_label_ds = path_ds.map(lambda x, y: load_and_preprocess_img(x, y, [256,256]), num_parallel_calls=AUTOTUNE)

# python logic here
def arbitrary_python_logic(img):
    seq = iaa.Sequential([
          iaa.Affine(translate_percent={"x": np.random.uniform(-0.2, 0.2), "y": np.random.uniform(-0.2, 0.2)})
    ])

    img_aug = seq.augment_image(img)

    return img_aug

# online image augmetation
def image_augmentation(img, label):
    img_aug = tf.image.random_flip_left_right(img)
    img_aug = tf.py_function(arbitrary_python_logic, [img_aug], (tf.float32))

    return img_aug, label

# Also use map() to perform augmentation on each sample
image_label_ds = image_label_ds.map(lambda x, y: image_augmentation(x, y), AUTOTUNE)

# Add all the settings
#image_label_ds = image_label_ds.shuffle(3) # we want to see the same cat for demo purpose
image_label_ds = image_label_ds.batch(2)
image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)

# iterate through the dataset
for _ in range(3):
    for image, label in image_label_ds:
        plt.figure()
        plt.imshow(image.numpy()[0]) # plot the first sample in batch
        plt.title(label.numpy()[0])