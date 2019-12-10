# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 01:29:15 2019

@author: Wei-Hsiang, Shen
"""

import tensorflow as tf
from tensorflow.keras import layers


'''
Image classification with 10 categories
'''
def ResNet():
    # Use tf.keras.Input to initialize the input tensor
    input_tensor = tf.keras.Input(shape=(32,32,3), batch_size=None) # batch size not yet defined

    x = layers.Conv2D(32, 3, activation='relu')(input_tensor)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output]) # element-wise tensor addition

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    # Use tf.keras.Model and specify the inputs and ouputs tensor
    model = tf.keras.Model(inputs=[input_tensor], outputs=[output_tensor])

    return model

if __name__ == '__main__':
    model = ResNet()

    # Plot and inspect the model
    model.summary()
    tf.keras.utils.plot_model(model, '1_model.png', show_shapes=True)