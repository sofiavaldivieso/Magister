import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import numpy as np


def load_data(base_dir, IMG_SIZE, seed, metodologia):
    BATCH_SIZE = 16
    if metodologia=='Metodologia 1' or metodologia=='Metodologia 2':
        train_dataset = image_dataset_from_directory(f'{base_dir}/train',
                                                 shuffle=True,
                                                 seed= seed,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
        validation_dataset = image_dataset_from_directory(f'{base_dir}/val',
                                                  shuffle=True,
                                                  seed= seed,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
    
    else:
        train_dataset = image_dataset_from_directory(base_dir,
                                                 shuffle=True,
                                                 validation_split=0.3,
                                                 subset="training",
                                                 seed= seed,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)

    validation_dataset = image_dataset_from_directory(base_dir,
                                                      shuffle=True,
                                                      validation_split=0.3,
                                                      subset="validation",
                                                      seed= seed,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)
    
    class_names = train_dataset.class_names
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, class_names


def load_data_fusion(base_dir, IMG_SIZE, seed):
    train_dataset, validation_dataset, class_names = load_data(base_dir, IMG_SIZE, seed)
    return train_dataset, test_dataset, validation_dataset, class_names

