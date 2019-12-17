# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:11:52 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import tensorflow as tf
import os
import imgaug.augmenters as iaa
import random

from tensorflow.keras import layers


def Get_Dataset(BATCH_SIZE):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def load_and_preprocess_img(path, label, shape):
        image_string = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, shape)
        image = image / 255.0

        return image, label

    def sequential_img_aug(img):
        seq = iaa.Sequential([
              iaa.Affine(translate_percent={"x": np.random.uniform(-0.2, 0.2), "y": np.random.uniform(-0.2, 0.2)})
        ])

        img_aug = seq.augment_image(img)

        return img_aug

    def image_augmentation(img, label):
        img_aug = tf.image.random_flip_left_right(img)
        img_aug = tf.py_function(sequential_img_aug, [img_aug], (tf.float32))

        return img_aug, label

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
    # notice the filepath order in linux

    print('Total samples:', len(label_list))
    print('Cat samples:', label_list.count(0))
    print('Dog samples:', label_list.count(1))

    # Shuffle the list (Python magic)
    temp = list(zip(img_path_list, label_list))
    random.shuffle(temp)
    img_path_list, label_list = zip(*temp)
    img_path_list, label_list = list(img_path_list), list(label_list)

    # Random train/val split
    X_train = img_path_list[:20000]
    y_train = label_list[:20000]
    X_val = img_path_list[20000:]
    y_val = label_list[20000:]

    # Construct tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(lambda x, y: load_and_preprocess_img(x, y, [256,256]), AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.map(lambda x, y: image_augmentation(x, y), AUTOTUNE) # augment only the training dataset
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.map(lambda x, y: load_and_preprocess_img(x, y, [256,256]), AUTOTUNE)

    # Add all the settings
    train_ds = train_ds.shuffle(2048)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    train_count = 20000
    val_count = 5000

    print('Training samples:', train_count)
    print('Validation samples:', val_count)

    return train_ds, val_ds, train_count, val_count

def Net():
    model = tf.keras.Sequential([
            layers.Conv2D(32, (3,3), (2,2), activation='relu', input_shape=(256, 256, 3)),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(64, (3,3), (2,2), activation='relu'),
            layers.MaxPool2D((2,2)),
            layers.Conv2D(128, (3,3), (2,2), activation='relu'),
            layers.MaxPool2D((2,2)),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

    return model

if __name__ == '__main__':
    BATCH_SIZE = 32

    model = Net()
    train_ds, val_ds, train_count, val_count = Get_Dataset(BATCH_SIZE)

    # callbacks
    checkpoint_path = "./checkpoints/net_weights_{epoch:02d}.h5"
    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True,
                                                     save_freq='epoch')
    csv_logger = tf.keras.callbacks.CSVLogger('./checkpoints/training.log')

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # train the model
    model.fit(x = train_ds.repeat(),
              validation_data = val_ds.repeat(),
              epochs = 10,
              verbose = 1,
              steps_per_epoch = train_count//BATCH_SIZE,
              validation_steps = val_count//BATCH_SIZE,
              callbacks=[save_checkpoint, csv_logger])