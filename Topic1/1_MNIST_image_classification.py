# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:09:11 2019

@author: Wei-Hsiang, Shen
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers


# Check version
print('Version: ', tf.__version__)
print('Eager execution: ', tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# Download the MNIST dataset
mnist = tf.keras.datasets.mnist # keras built-in dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Plot some images
plt.figure()
plt.imshow(train_images[0,:,:], cmap='gray')
plt.colorbar()
plt.title(train_labels[0])

plt.figure()
plt.imshow(test_images[0,:,:], cmap='gray')
plt.colorbar()
plt.title(test_labels[0])

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Build the model using Sequential API
model = tf.keras.Sequential()
model.add(layers.Conv2D(8, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Flatten()) # add the layers one-by-one
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Inspect the model (check the parameters/shape/graph)
model.summary()
tf.keras.utils.plot_model(model, to_file='MNIST_model.png', show_shapes=True)

# Compile the model (determine optimizer and loss)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# Set-up training callbacks
checkpoint_path = "./checkpoints/net_MNIST_weights_{epoch:02d}.h5"
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)
csv_logger = tf.keras.callbacks.CSVLogger('./checkpoints/training_MNIST.txt')

# Train the model
history = model.fit(x=train_images,
                    y=train_labels,
                    validation_data=(test_images, test_labels),
                    batch_size=32,
                    epochs=10,
                    callbacks=[save_checkpoint, csv_logger])

# Plot training progress
print(history.history.keys()) # check what it stores for us

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('loss')

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('accuracy')

# Evaluate (test) the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss: {:.4f}'.format(test_loss))
print('Test acc: {:.4f}'.format(test_acc))

# Model inference (prediction)
img = test_images[100,:,:] # img = test_images[100]

img = np.expand_dims(img, axis=0)
prediction = model(img) # eager prediction
prediction = model.predict(img) # compile then predict

predicted_class = np.argmax(prediction)

plt.figure()
plt.imshow(img[0], cmap='gray')
plt.title(predicted_class)