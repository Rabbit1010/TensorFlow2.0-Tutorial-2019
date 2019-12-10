# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:13:52 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers


def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False))

    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result

def Net():
    input_tensor = tf.keras.layers.Input(shape=[256,256,3])

    # a list of down sample blocks
    down_stack = [
        downsample(64, 4),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
    ]

    # a list of up sample blocks
    up_stack = [
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    # Downsampling through the model
    x = input_tensor
    skips = []
    for down_block in down_stack:
        x = down_block(x)
        skips.append(x)

    # Upsampling and establishing the skip connections
    for up_block, skip_tensor in zip(up_stack, reversed(skips[:-1])):
        x = up_block(x)
        x = layers.Concatenate(axis=-1)([x, skip_tensor])

    x = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(x)

    return tf.keras.Model(inputs=input_tensor, outputs=x)

if __name__ == '__main__':
    model = Net()

    # Plot and inspect the model
    model.summary()
    tf.keras.utils.plot_model(model, '4_UNet.png', show_shapes=True)