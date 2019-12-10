#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:09:50 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


# Conv2D + BatchNorm + ReLU
def Vision_Block(n_filters, strides=(1,1), name=None):
    block = tf.keras.Sequential(name=name)
    block.add(layers.Conv2D(n_filters, kernel_size=(3,3), strides=strides, padding='same', use_bias=False))
    block.add(layers.BatchNormalization())
    block.add(layers.ReLU())

    return block

def Base_Net():
    input_tensor = tf.keras.Input(shape=(300, 160, 3))

    # First, go through vision blocks to extract image features
    x = input_tensor
    for i in range(4):
        x = Vision_Block(32*2**i, (2,2), name='Vision_Block_{}'.format(i))(x)

    conv_shape = x.get_shape()
    x = layers.Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
    x = layers.Dense(32, activation='relu')(x) # (bs, 19, 32)

    # Second, use RNN to generate output sequence
    rnn_size = 128
    gru_1 = layers.GRU(rnn_size, return_sequences=True)(x)
    gru_1b = layers.GRU(rnn_size, return_sequences=True, go_backwards=True)(x)
    gru1_merged = layers.Add()([gru_1, gru_1b])

    gru_2 = layers.GRU(rnn_size, return_sequences=True)(gru1_merged)
    gru_2b = layers.GRU(rnn_size, return_sequences=True, go_backwards=True)(gru1_merged)
    x = layers.Concatenate()([gru_2, gru_2b])

    x = layers.Dropout(0.25)(x)
    x = layers.Dense(27, activation='softmax')(x)

    return tf.keras.Model(inputs=input_tensor, outputs=x)

def Whole_Net():
    def ctc_loss_func(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    input_tensor = tf.keras.Input(shape=(300, 160, 3))
    base_model = Base_Net()

    x = base_model(input_tensor)

    labels = tf.keras.Input(name='the_labels', shape=(5,), dtype='float32')
    input_length = tf.keras.Input(name='input_length', shape=(1,), dtype='int64')
    label_length = tf.keras.Input(name='label_length', shape=(1,), dtype='int64')

    loss_out = layers.Lambda(ctc_loss_func, output_shape=(1,),
                             name='ctc')([x, labels, input_length, label_length])

    whole_model = tf.keras.Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])

    whole_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')

    return whole_model, base_model

if __name__ == '__main__':
    whole_model, base_model = Whole_Net()

    base_model.summary()
    tf.keras.utils.plot_model(base_model, '5_base_model.png', show_shapes=True)

    whole_model.summary()
    tf.keras.utils.plot_model(whole_model, '5_whole_model.png', show_shapes=True)
