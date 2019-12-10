# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 03:11:08 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers


'''
The model takes in two images of different size and classify it into 10 categories
and predict whether it is the same viewpoint.
'''
def Net():
    # Define the convolutional layers for feature extraction
    image_input = tf.keras.Input(shape=(None, None, 3)) # None means that it is not yet defined
    x = layers.Conv2D(2, (3, 3))(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(4, (3, 3))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    out = x

    # Here, we give the model a name.
    feature_extractor = tf.keras.Model(image_input, out, name='Image_Feature_Extractor')

    # Then define the classification model
    image_1 = tf.keras.Input(shape=(30, 30, 3), name='Image_1')
    image_2 = tf.keras.Input(shape=(60, 60, 3), name='Image_2')

    # The feature extractor CNN model will be shared, weights and all
    out_a = feature_extractor(image_1) # models are callable like layers
    out_b = feature_extractor(image_2)

    out_a = layers.Flatten()(out_a)
    out_b = layers.Flatten()(out_b)

    concatenated = layers.concatenate([out_a, out_b], axis=-1)

    category_classification = layers.Dense(10, activation='softmax', name='category')(concatenated)
    viewpoint_prediction = layers.Dense(1, activation='tanh', name='viewpoint')(concatenated)

    # specify the input and outputs tensors
    main_model = tf.keras.Model(inputs=[image_1, image_2],
                                outputs=[category_classification, viewpoint_prediction])

    # two outputs have two loss function to optimize
    main_model.compile(optimizer='rmsprop',
                        # specify loss function by output name (or in order)
                       loss={'category': 'sparse_categorical_crossentropy',
                             'viewpoint': 'binary_crossentropy'},
                             # specify the weighting of each loss
                             loss_weights=[1., 0.2])

    return main_model

if __name__ == '__main__':
    model = Net()

    # Plot and inspect the model
    model.summary()
    tf.keras.utils.plot_model(model, '3_model.png', show_shapes=True)