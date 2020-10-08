"""
author: Sanidhya Mangal
github: sanidhyamangal
"""

import os  # for os relates ops
import time  # for time related ops
from functools import partial  # for making base model layers

import matplotlib.pyplot as plt  # for plotting the images
import tensorflow as tf  # for deep learning

from data_handler import (
    data_loader_csv_unsupervisied,  # data loader
    generate_noise)

train_path = "./fashion-mnist_train.csv"

train_dataset = data_loader_csv_unsupervisied(train_path)


def generate_save_plot(x: tf.Tensor, model: tf.keras.Model, image_name, n=8):
    plt.figure(figsize=(4, 4))
    for i in range(n):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.squeeze(x[i]))
        plt.gray()
        plt.axis('off')
    for i in range(n):
        plt.subplot(4, 4, i + n + 1)
        plt.imshow(tf.squeeze(model(x)[i]))
        plt.gray()
        plt.axis('off')
    plt.savefig(image_name)


# create a class for AutoEncoderClass
class DenoiseAutoEncoder(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(DenoiseAutoEncoder, self).__init__(*args, **kwargs)

        # base conv2d layer
        self.conv2d = partial(tf.keras.layers.Conv2D,
                              kernel_size=(3, 3),
                              padding='same',
                              activation=tf.nn.relu,
                              strides=2)

        # base conv2d transpose layer
        self.conv2d_transpose = partial(tf.keras.layers.Conv2DTranspose,
                                        kernel_size=(3, 3),
                                        padding='same',
                                        strides=2)

        # encoder layer
        self.encoder = tf.keras.Sequential(
            layers=[self.conv2d(
                filters=16), self.conv2d(filters=8)])

        # decoder layer
        self.decoder = tf.keras.Sequential([
            self.conv2d_transpose(filters=8),
            self.conv2d_transpose(filters=16),
            tf.keras.layers.Conv2DTranspose(filters=1,
                                            kernel_size=3,
                                            padding="same",
                                            activation=tf.nn.sigmoid)
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# define autoencoder instance
autoencoder = DenoiseAutoEncoder()

# define loss function
mseloss = tf.losses.MeanSquaredError()

# define optimizer
optimizer = tf.keras.optimizers.Adam()

# create checkpoint functions
# create a checkpoint for the models
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(autoencoder=autoencoder, optimizer=optimizer)


@tf.function
def train_step(model, noisy_image, true_images):
    with tf.GradientTape() as tape:
        loss = mseloss(true_images, model(noisy_image))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


test_path = "./fashion-mnist_test.csv"
for test_data in data_loader_csv_unsupervisied(test_path).take(1):
    test_data_noisy = generate_noise(test_data)

# training loop
EPOCHS = 10
for i in range(1, EPOCHS + 1):

    start_time = time.time()  # start time for the epoch
    for data in train_dataset:

        # train step
        train_step(autoencoder, generate_noise(data), data)

    generate_save_plot(data, autencoder, f"{i}.png")

    if (i % 5) == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print("Training Time for Epoch {} : {}".format(i,
                                                   time.time() - start_time))
