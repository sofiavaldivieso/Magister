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

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(fp, tp, label=name, linewidth=1.5, **kwargs)
    plt.xlabel('False positives')
    plt.ylabel('True positives')
    plt.axis([-0.01,1.01,-0.01,1.01]) 
    plt.grid(True)
    
def plot_cm(labels, predictions, plots_path, i, p=0.5):
    class_names= ['ARE', 'NORMAL']
    cm = confusion_matrix(labels, predictions > p)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(df, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    fig.savefig(f'{plots_path}/CM'+str(i)+'.png')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total: ', np.sum(cm[1],cm[0]))


def plot_metrics(history, plots_path, i, fine_tunning=False, history_fine=None):
    fig = plt.figure(figsize=(15, 15))
    metrics = ['loss', 'prc', 'precision', 'recall']
    if fine_tunning:
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch + history_fine.epoch, history.history[metric] + history_fine.history[metric],
                     color='blue', label='Train')
            plt.plot(history.epoch + history_fine.epoch, history.history['val_' + metric] + history_fine.history['val_' + metric],
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
    fig.savefig(f'{plots_path}/Metrics_'+str(i)+'.png')
            


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    
