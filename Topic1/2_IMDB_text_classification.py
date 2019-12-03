# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:09:11 2019

@author: Wei-Hsiang, Shen
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers


# Download the IMDB dataset
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=5000)
word_index = imdb.get_word_index()

# -------------------------------------------
# Consctruct decoder and encoder (some Python magic here)
# -------------------------------------------
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0 # The first indices are reserved
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def encode_review(string):
    out = []
    for word in string.split(' '):
        try:
            out.append(word_index[word])
        except KeyError:
            out.append(word_index["<UNK>"])
    return np.array(out)
# -------------------------------------------

# Show some data
print(test_data[0])
print(decode_review(test_data[0]))
print('Label:', test_labels[0])

# Preprocess (pad all sequence to the same length)
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# Build the model using Sequential API
model = tf.keras.Sequential([
            layers.Embedding(input_dim=5000, output_dim=16, input_length=256),
            layers.GlobalAveragePooling1D(),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

# Inspect the model (check the parameters/shape/graph)
model.summary()
tf.keras.utils.plot_model(model, to_file='IMDB_model.png', show_shapes=True)

# Compile the model (determine optimizer and loss)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# Set-up callbacks
checkpoint_path = "./checkpoints/net_IMDB_weights_{epoch:02d}.h5"
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)
csv_logger = tf.keras.callbacks.CSVLogger('./checkpoints/training_IMDB.txt')

# Train the model
history = model.fit(x=train_data,
                    y=train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose=2,
                    callbacks=[save_checkpoint, csv_logger])

# Plot training progress
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

# Model inference (prediction)
input_tensor = encode_review('This movie is really awful and bad, and I think it is a waste of money')
decode_review(input_tensor) # check encoded text
input_tensor = np.expand_dims(input_tensor, axis=0) # create a batch axis
input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                             value=word_index["<PAD>"],
                                                             padding='post',
                                                             maxlen=256)

prediction = model(input_tensor)
print(prediction.numpy())