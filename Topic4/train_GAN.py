#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:11:08 2019

@author: Wei-Hsiang, Shen
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import time

from GAN_model import Generator_Model, Discriminator_Model
from data_generator import Get_DS


EPOCHS = 1000
BATCH_SIZE = 128 # this should be much smaller in generative models

train_ds, val_ds = Get_DS(BATCH_SIZE)

generator = Generator_Model()
discriminator = Discriminator_Model()

noise_dim = 200 # input shape of the generator

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(judgment_real, judgment_fake):
    # judgement from real images should be close to 1
    real_loss = cross_entropy(tf.ones_like(judgment_real), judgment_real)
    
    # judgement from fake images should be close to 0
    fake_loss = cross_entropy(tf.zeros_like(judgment_fake), judgment_fake)
    
    # Total loss 
    total_loss = real_loss + fake_loss
    
    return total_loss

def generator_loss(judgment_fake):
    # generator wants the judgment of fake images to be close to 1
    return cross_entropy(tf.ones_like(judgment_fake), judgment_fake)

# We will reuse this seed overtime (so it's easier) to visualize progress
num_examples_to_generate = 25
seed = tf.random.normal([num_examples_to_generate, noise_dim])

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False, so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    predictions = predictions.numpy()
    predictions[predictions<0] = 0
    predictions[predictions>1] = 1

    fig = plt.figure(dpi=100, figsize=(16, 16))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i], vmin=0, vmax=1)
        plt.axis('off')

    plt.savefig('./results/test_image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)

generator_optimizer = tf.keras.optimizers.RMSprop(1e-4)
discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-4)

@tf.function
def train_step(batch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # we want to have two tapes so that we can get two different gradients
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Notice `training` is set to True, so all layers run in training mode (batchnorm).
        
        # Generator generates a batch of images from noise
        generated_images = generator(noise, training=True)

        # Discriminator takes in fake images from generator and true images from dataset
        judgment_real = discriminator(batch, training=True)
        judgment_fake = discriminator(generated_images, training=True)

        # calculate the loss of both models
        gen_loss = generator_loss(judgment_fake)
        disc_loss = discriminator_loss(judgment_real, judgment_fake)

    # Get their graident from loss to all variables
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradient descent using optimizer
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Main training loop
for epoch in range(EPOCHS):
    start_time = time.time()

    gen_loss_total = 0
    disc_loss_total = 0
    i_step = 1

    print("Epoch {}/{}".format(epoch+1, EPOCHS))
    for batch in train_ds: # for each batch, note that we do not code batch per epoch, the dataset would end if all data is used
        gen_loss, disc_loss = train_step(batch)
        gen_loss_total += gen_loss
        disc_loss_total += disc_loss
        
        print("\rStep {}".format(i_step), end='')
        i_step += 1

    end_time = time.time()        
    print(', gen_loss: {:.3f}, disc_loss: {:.3f}, time: {:.2f} sec'.format(gen_loss_total, disc_loss_total, end_time-start_time))
    
    if epoch%50 == 0: # save weights every 50 epochs
        generator.save_weights('./checkpoints/generator_weights_{}.h5'.format(epoch))
        
    # Save generated image at the end of each epoch
    generate_and_save_images(generator, epoch + 1, seed)