import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(base_dir, strategy):
    BATCH_SIZE_PER_REPLICA = 16
    batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'val')
    train_datagen = ImageDataGenerator(rotation_range=40,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               rescale=1.0 / 255,
                               horizontal_flip=True,
                               fill_mode='nearest', 
                               validation_split=0.04)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    classes = ['NORMAL', 'AR']
    train_batches = train_datagen.flow_from_directory(train_dir,
                                              target_size=(160, 160),
                                              classes=classes,
                                              subset='training',
                                              shuffle=True,
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              color_mode='rgb')
    test_batches = test_datagen.flow_from_directory(test_dir,
                                            target_size=(160, 160),
                                            classes=classes,
                                            shuffle=True,
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            color_mode='rgb')
    val_batches = train_datagen.flow_from_directory(train_dir,
                                           subset='validation',
                                           target_size=(160, 160),
                                           classes=classes,
                                           shuffle=True,
                                           batch_size=batch_size,
                                           class_mode='categorical',
                                           color_mode='rgb')
    return train_batches, test_batches, val_batches, classes

