# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:59:22 2022

@author: tommyshen
"""

import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects
from tensorflow.nn import depth_to_space

class SubpixelConv2D(Layer):
    """ Subpixel Conv2D Layer
    upsampling a layer from (h, w, c) to (h*r, w*r, c/(r*r)),
    where r is the scaling factor, default to 4
    # Arguments
    upsampling_factor: the scaling factor
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        the second and the third dimension increased by a factor of
        `upsampling_factor`; the last layer decreased by a factor of
        `upsampling_factor^2`.
    # References
        Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network Shi et Al. https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upsampling_factor=4, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                             'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs, **kwargs):
        return depth_to_space( inputs, self.upsampling_factor )

    def get_config(self):
        config = { 'upsampling_factor': self.upsampling_factor, }
        base_config = super(SubpixelConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [ input_shape[0],
                 input_shape_1,
                 input_shape_2,
                 int(input_shape[3]/factor)
               ]
        return tuple( dims )

get_custom_objects().update({'SubpixelConv2D': SubpixelConv2D})

if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    
    ip = Input(shape=(2, 2, 4))
    x = SubpixelConv2D(upsampling_factor=2)(ip)
    
    model = Model(ip, x)
    model.summary()

    # %% test run the model to check pixel shuffle effects
    import numpy as np
    a = np.ones((2,2,4))
    
    i_c = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                a[i,j,k] = i_c
                i_c += 1
    
    input_tensor = tf.constant(a)
    print(input_tensor.shape)
    
    output_tensor = model(np.expand_dims(a,axis=0))
    
    out = output_tensor.numpy()[0]