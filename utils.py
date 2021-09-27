import tensorflow as tf
import keras
from keras import callbacks
from Modelos_TL_codes.load_data import load_data


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
import numpy as np

from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

def load_data_train(base_dir, IMG_SIZE):
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'val')
    BATCH_SIZE = 128
    train_dataset = image_dataset_from_directory(train_dir,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    class_names = train_dataset.class_names

    test_dataset = image_dataset_from_directory(test_dir,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, test_dataset, class_names

def plot_roc(name, labels, predictions, plots_path, i, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    plt.savefig(f'{plot_path}/roc'+str(i)+'.png')



def plot_cm(labels, predictions, plots_path, i, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(f'{plot_path}/CM'+str(i)+'.png')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total: ', np.sum(cm[1]))


def plot_metrics(history, plots_path, i, fine_tunning=False, history_fine=None):
    plt.figure(figsize=(8, 8))
    metrics = ['loss', 'prc', 'precision', 'recall']
    if fine_tunning:
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch + history_fine.epoch, history.history[metric] + history_fine.history[metric],
                     color='blue', label='Train')
            plt.plot(history.epoch, history.history['val_' + metric] + history_fine.history['val_' + metric],
                     color='blue', linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()
            plt.savefig(f'{plot_path}/metrics'+str(i)+'.png')
    else:
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
            plt.plot(history.epoch, history.history['val_' + metric],
                     color='blue', linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()
            plt.savefig(f'{plot_path}/metrics'+str(i)+'.png')
            


def plot_prc(name, labels, predictions, plots_path, i, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(f'{plot_path}/prc'+str(i)+'.png')



def visualize(history, baseline_results, model, base_dir, IMG_SIZE, BATCH_SIZE,
              fine_tunning=False, history_fine=None):
    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'prc', 'precision', 'recall']
    train_dataset, test_dataset, class_names = load_data_train(base_dir, IMG_SIZE)
    image_batch_train, train_labels = next(iter(train_dataset))
    image_batch_test, test_labels = next(iter(test_dataset))
    train_predictions = model.predict(image_batch_train, batch_size=BATCH_SIZE)
    test_predictions = model.predict(image_batch_test, batch_size=BATCH_SIZE)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    plot_cm(test_labels, test_predictions)

    plot_roc("Train", train_labels, train_predictions, color=colors[0])
    plot_roc("Test", test_labels, test_predictions, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')

    plot_prc("Train", train_labels, train_predictions, color=colors[0])
    plot_prc("Test", test_labels, test_predictions, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')

    plot_metrics(history, fine_tunning, history_fine)
