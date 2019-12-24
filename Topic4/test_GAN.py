#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 00:37:05 2019

@author: Wei-Hsiang, Shen
"""


import matplotlib.pyplot as plt
import tensorflow as tf

from GAN_model import Generator_Model

noise = tf.random.normal([1, 200])

generator = Generator_Model()
generator.load_weights('./checkpoints/generator_weights_100.h5')

generated_img = generator.predict(noise)

plt.figure()
plt.imshow(generated_img[0], vmin=0, vmax=1)

# Random walk on the noise
walk_direction = tf.random.normal([1,200])

noise_walk = noise
for _ in range(10):
    noise_walk = noise_walk + walk_direction/3
    
    generated_img = generator.predict(noise_walk)
    
    plt.figure()
    plt.imshow(generated_img[0], vmin=0, vmax=1)