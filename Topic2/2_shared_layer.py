# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 01:29:15 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers


'''
The model takes in two sequences of texts, and predicts whether the comments are
from the same person.
'''
def Net():
    comment_a = tf.keras.Input(shape=(256,), name='Comment_a')
    comment_b = tf.keras.Input(shape=(256,), name='Comment_b')

    # The two comments go through the same embedding and feature extraction
    # Initialize the layer instance
    shared_embedding = layers.Embedding(input_dim=5000, output_dim=16)
    shared_lstm = layers.LSTM(64, return_sequences=True)

    # When calling the layer instance multiple times, the weights are reused.
    # (It's the same layer instance)
    embedded_a = shared_embedding(comment_a)
    embedded_b = shared_embedding(comment_b)
    encoded_a = shared_lstm(embedded_a)
    encoded_b = shared_lstm(embedded_b)

    # Then, the encoded vectors go through two different GRU
    encoded_a = layers.GRU(32)(encoded_a)
    encoded_b = layers.GRU(32)(encoded_b)

    # We can then concatenate the two vectors:
    merged_vector = layers.concatenate([encoded_a, encoded_b], axis=-1)

    predictions = layers.Dense(1, activation='tanh')(merged_vector)

    # Model with multiple inputs
    model = tf.keras.Model(inputs=[comment_a, comment_b], outputs=predictions)

    return model

if __name__ == '__main__':
    model = Net()

    # Plot and inspect the model
    model.summary()
    tf.keras.utils.plot_model(model, '2_model.png', show_shapes=True)