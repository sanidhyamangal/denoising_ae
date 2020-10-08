"""
Author: Sanidhya Mangal
github: sanidhyamangal
"""
import glob  # for glob based operations
import pathlib  # for path based ops

import pandas as pd  # for data frame based ops
import tensorflow as tf  # for deep learning and data processing

train_path = "/home/sanidhya/Dataset/fashion-mnist_train.csv"


def generate_noise(x: tf.Tensor,
                   noise_rate: float = 0.2,
                   min_clip: float = 0.,
                   max_clip: float = 1.) -> tf.Tensor:
    return tf.clip_by_value(
        x + noise_rate * tf.random.normal(shape=x.shape, dtype=tf.float64),
        clip_value_min=min_clip,
        clip_value_max=max_clip)


def data_loader_csv_unsupervisied(df_path: str, batch_size=64, shuffle=True):

    data = pd.read_csv(df_path)

    data_set_data = tf.convert_to_tensor(data.iloc[:, 1:], tf.float64)

    # free up the memory for train set test_set
    del data

    # create a train and test dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (data_set_data)).shuffle(60000).batch(batch_size).map(
            lambda x: tf.reshape(x / 255.0, shape=[-1, 28, 28, 1]))

    del data_set_data

    # return dataset
    return dataset


# train_path = "/home/sanidhya/Dataset/fashion-mnist_train.csv"