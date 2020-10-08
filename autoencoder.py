"""
author: Sanidhya Mangal
github: sanidhyamangal
"""

import os  # for os relates ops

import matplotlib.pyplot as plt  # for plotting the images
import tensorflow as tf  # for deep learning

from data_handler import data_loader_csv_unsupervisied # data loader

train_path = "/home/sanidhya/Dataset/fashion-mnist_train.csv"

train_dataset = data_loader_csv_unsupervisied(train_path)

# function to generate noise in images
def generate_noise(x:tf.Tensor, noise_rate:float=0.2) -> tf.Tensor:
    return x + noise_rate * tf.random.normal(shape=x.shape)
