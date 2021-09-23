import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os


def load_data(base_dir, IMG_SIZE):
    BATCH_SIZE = 16
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'val')
    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    class_names = train_dataset.class_names

    validation_dataset = image_dataset_from_directory(test_dir,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)

    test_dataset = image_dataset_from_directory(test_dir,
                                                batch_size=128,
                                                image_size=IMG_SIZE)    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, test_dataset, validation_dataset, class_names


def load_data_fusion(base_dir, IMG_SIZE):
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'val')
    BATCH_SIZE = 16
    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=False,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    class_names = train_dataset.class_names

    validation_dataset = image_dataset_from_directory(test_dir,
                                                      shuffle=False,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)

    #test_dataset = image_dataset_from_directory(test_dir,
                                                #batch_size=BATCH_SIZE,
                                                #image_size=IMG_SIZE)
    
    train_batches = tf.data.experimental.cardinality(train_dataset)
    test_dataset = train_dataset.take(train_batches // 10)
    train_dataset = train_dataset.skip(train_batches // 10)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, test_dataset, validation_dataset, class_names

