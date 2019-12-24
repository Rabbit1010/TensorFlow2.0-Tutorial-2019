# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:32:46 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os

from tensorflow.keras import layers

# Training parameters
BATCH_SIZE = 128
TOTAL_EPOCHS = 10

# We use the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

train_images = np.array(train_images, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.int64)
test_images = np.array(test_images, dtype=np.float32)
test_labels = np.array(test_labels, dtype=np.int64)

# Construct the data input pipeline
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_ds = train_ds.shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Build a simple CNN
model = tf.keras.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
    ])


@tf.function # with this function decorator, tensorflow compiles the function into graph
def train_step(model, batch, optimizer):
    with tf.GradientTape() as tape: # tell the gradient tape to record the gradient
        pred = model(batch[0])
        target = batch[1]

        # calculate loss
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(target, pred)

    # Get the gradient using tf.GradientTape().gradient
    gradients = tape.gradient(loss, model.trainable_variables) # tape is released as soon as gradient() is called
    
    # Let the optimizer apply the gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

@tf.function # with this function decorator, tensorflow compiles the function into graph
def val_step(model, batch):
    pred = model(batch[0])
    truth = batch[1]

    pred = tf.argmax(pred, axis=-1)

    hit = tf.equal(pred, truth)
    hit_count = tf.reduce_sum(tf.cast(hit, tf.float32)) # number of 'True'

    return hit_count

opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
history = {'loss':[], 'val_acc':[]}

# Main training loop
for epoch in range(TOTAL_EPOCHS):
    start_time = time.time()
    print("Epoch {}/{}".format(epoch+1, TOTAL_EPOCHS))

    i_step = 1

    # Optimize the model using the training data
    loss_total = 0
    for batch in train_ds:
        loss = train_step(model, batch, opt)

        print("\rStep {}, Loss: {:.3f}".format(i_step, loss), end='')

        loss_total += loss
        i_step += 1

    train_loss = loss_total / (60000//BATCH_SIZE)

    # Evaluate on the validation data
    val_hit_total = 0
    for batch in val_ds:
        hit = val_step(model, batch)
        val_hit_total += hit

    val_acc = val_hit_total / 10000

    end_time = time.time()
    
    # Save model weights
    model.save_weights('./checkpoints/net_weights_{}.h5'.format(epoch))

    # Store training histroy
    history['loss'].append(train_loss.numpy())
    history['val_acc'].append(val_acc.numpy())

    print(", Val_acc: {:.3f}".format(val_acc), end='')
    print(", Time: {:.2f} sec".format(end_time-start_time))

# print(history)