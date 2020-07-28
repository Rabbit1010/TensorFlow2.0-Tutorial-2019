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

BATCH_SIZE = 128
train_ds = train_ds.shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Build a simple CNN
model = tf.keras.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
    ])

TOTAL_EPOCHS = 10

@tf.function # with this function decorator, tensorflow compiles the function into graph
def train_step(model, batch, optimizer):
    with tf.GradientTape() as tape: # tell the gradient tape to record the gradient
        pred = model(batch[0], training=True) # some layers have different operations in training and inference time
        target = batch[1]

        # calculate loss
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(target, pred)

    # Get the gradient using tf.GradientTape().gradient
    gradients = tape.gradient(loss, model.trainable_variables) # the gradient is released as soon as gradient() is called
    # Let the optimizer apply the gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Calculate training accuracy
    pred = model(batch[0])
    pred = tf.argmax(pred, axis=-1)
    truth = batch[1]

    hit = tf.equal(pred, truth)
    hit_count = tf.reduce_sum(tf.cast(hit, tf.float32)) # number of 'True

    return loss, hit_count

@tf.function # with this function decorator, tensorflow compiles the function into graph
def val_step(model, batch):
    pred = model(batch[0])
    truth = batch[1]
    val_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(truth, pred)

    pred = tf.argmax(pred, axis=-1)

    hit = tf.equal(pred, truth)
    hit_count = tf.reduce_sum(tf.cast(hit, tf.float32)) # number of 'True'

    return val_loss, hit_count

opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
history = {'epoch':[], 'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}

# Main training loop
for epoch in range(TOTAL_EPOCHS):
    start_time = time.time()
    print("Epoch {}/{}".format(epoch+1, TOTAL_EPOCHS))

    i_step = 1

    # Optimize the model using the training data
    loss_total = 0
    hit_total = 0
    for batch in train_ds:
        loss, hit = train_step(model, batch, opt)

        print("\rStep {}, Loss: {:.3f}, Train_acc: {:.4f}".format(i_step, loss, hit/batch[0].shape[0]), end='')

        loss_total += loss
        hit_total += hit
        i_step += 1

    train_loss = loss_total / (60000//BATCH_SIZE)
    train_acc = hit_total / 60000

    # Evaluate on the validation data
    val_loss_total = 0 
    val_hit_total = 0    
    for batch in val_ds:
        loss, hit = val_step(model, batch)
        val_hit_total += hit
        val_loss_total += loss

    val_loss = val_loss_total / (10000//BATCH_SIZE)
    val_acc = val_hit_total / 10000

    end_time = time.time()

    # Save model weights
    model.save_weights('./checkpoints/net_weights_{}.h5'.format(epoch))

    # Store training histroy
    history['epoch'].append(epoch)
    history['loss'].append(train_loss.numpy())
    history['acc'].append(train_acc.numpy())
    history['val_loss'].append(val_loss.numpy())
    history['val_acc'].append(val_acc.numpy())

    print(", Val_loss: {:.3f}, Val_acc: {:.3f}".format(val_loss, val_acc), end='')
    print(", Time: {:.2f} sec".format(end_time-start_time))

# Plot the training history
plt.figure()
plt.plot(history['epoch'], history['loss'])
plt.plot(history['epoch'], history['val_loss'])
plt.legend(['train', 'val'])

plt.figure()
plt.plot(history['epoch'], history['acc'])
plt.plot(history['epoch'], history['val_acc'])
plt.legend(['train', 'val'])
