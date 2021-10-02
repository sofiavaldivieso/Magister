import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import numpy as np


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


def make_pairs(OCT_dir, FUNDUS_dir, IMG_SIZE):
  pairImages = []
  pairLabels = []
  train_dataset_OCT, test_dataset_OCT, validation_dataset_OCT, class_names_OCT =load_data_fusion(OCT_dir, IMG_SIZE)
  train_dataset_FUNDUS, test_dataset_FUNDUS, validation_dataset_FUNDUS, class_names_FUNDUS = load_data_fusion(FUNDUS_dir, IMG_SIZE)
  image_batch_train_OCT, train_labels_OCT = next(iter(train_dataset_OCT))
  image_batch_train_FUNDUS, train_labels_FUNDUS = next(iter(train_dataset_FUNDUS))
  image_batch_test_FUNDUS, test_labels_FUNDUS = next(iter(test_dataset_FUNDUS))
  image_batch_test_OCT, test_labels_OCT = next(iter(test_dataset_OCT))
  for i in range(len(image_batch_train_OCT)):
    img1= image_batch_train_OCT[i]
    img2= image_batch_train_FUNDUS[i]
    label= train_labels[i].numpy()
    if np.array_equal(img1,img2):
      pairImages.append([img1, img2])
      pairLabels.append([label])
    return (np.array(pairImages), np.array(pairLabels))
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

    test_dataset = image_dataset_from_directory(test_dir,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, test_dataset, validation_dataset, class_names

