import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import numpy as np


def load_data(base_dir, IMG_SIZE, seed):
    BATCH_SIZE = 16
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'val')
    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 validation_split=0.3,
                                                 subset="training",
                                                 seed= seed,
                                                 label_mode='binary',
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    class_names = train_dataset.class_names

    validation_dataset = image_dataset_from_directory(test_dir,
                                                      shuffle=True,
                                                      validation_split=0.3,
                                                      subset="validation",
                                                      seed= seed,
                                                      label_mode='binary',
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, class_names


def load_data_fusion(base_dir, IMG_SIZE, seed):
    train_dataset, validation_dataset, class_names = load_data(base_dir, IMG_SIZE, seed)
    return train_dataset, test_dataset, validation_dataset, class_names

