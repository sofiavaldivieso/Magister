


"""# Graficar

## Parametros
"""

import os
from Modelos_TL_codes.load_data import load_data
import tensorflow as tf
from Modelos_TL_codes.make_model import modelo, late_fusion
from utils import *
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


def load_test_data(base_dir, IMG_SIZE):
    BATCH_SIZE = 200
    test_dir = os.path.join(base_dir, 'test')
    train_dir = os.path.join(base_dir, 'train')
    train_dataset = image_dataset_from_directory(train_dir,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)
    test_dataset = image_dataset_from_directory(test_dir,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE)
    class_names = test_dataset.class_names

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, test_dataset, class_names



def graficar(Modelo, Metodologia):
    OCT_propio = '/content/drive/MyDrive/MAGISTER/DATA/DS_merged_OCT'
    OCT_FUNDUS_propio = '/content/drive/MyDrive/MAGISTER/DATA/DS_merged_OCT+FUNDUS'
    FUNDUS = '/content/drive/MyDrive/MAGISTER/DATA/Dataset propio 2021/Metodologia5/FUNDUS/'
    OCT_M5 = '/content/drive/MyDrive/MAGISTER/DATA/Dataset propio 2021/Metodologia5/OCT/'
    if Metodologia == "Metodologia 1" or Metodologia == "Metodologia 2":
        base_dir = OCT_propio
    elif Metodologia == "Metodologia 3" or Metodologia == "Metodologia 4":
        base_dir = OCT_FUNDUS_propio
    elif Metodologia == "Metodologia 5":
        base_dir = OCT_FUNDUS_propio
    else:
        base_dir = '/content/drive/MyDrive/MAGISTER/DATA/FUNDUS/ODIR/'

    history_path= os.path.join('/content/drive/MyDrive/MAGISTER/Codigos/Entrenamientos', Modelo, 'History', Metodologia)
    plots_path= os.path.join('/content/drive/MyDrive/MAGISTER/Codigos/Entrenamientos/', Modelo, 'Visualizar Metricas', Metodologia)
    tf.keras.backend.clear_session()
    #base_dir='/content/drive/MyDrive/MAGISTER/DATA/Dataset propio 2021/OCT+FUNDUS/output'     #poner path de OCT2017
    model_name= Modelo
    if Modelo =='Mobilenet':
      IMG_SIZE = (224, 224)
    else:
      IMG_SIZE = (150, 150)

    """## Crear el modelo"""
    metrics = [tf.keras.metrics.TruePositives(name='tp'),
               tf.keras.metrics.FalsePositives(name='fp'),
               tf.keras.metrics.TrueNegatives(name='tn'),
               tf.keras.metrics.FalseNegatives(name='fn'),
               tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall'),
               tf.keras.metrics.AUC(name='auc'),
               tf.keras.metrics.AUC(name='prc', curve='PR'),
               ]
    if Metodologia == 'Metodologia 5':
      model, base_model_OCT, base_model_FUNDUS = late_fusion(IMG_SIZE, Modelo, metrics, None, None)
    else:
      model, base_model = modelo(IMG_SIZE, Modelo, metrics)

    """## Checkpoint"""

    checkpoint_path = os.path.join('/content/drive/MyDrive/MAGISTER/Codigos/Checkpoints/', Modelo, Metodologia , 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(checkpoint_path)
    # Create a callback that saves the model's weights

    """## Visualizar metricas"""

    model.load_weights(checkpoint_path)
    if Metodologia == 'Metodologia 5':
      train_ds, test_ds, class_names = load_test_data(base_dir, (248, 632))
    else:
      train_ds, test_ds, class_names = load_test_data(base_dir, IMG_SIZE)
    print('Datos Cargados')
    BATCH_SIZE=200
    baseline_results = model.evaluate(test_ds,
                                      batch_size=BATCH_SIZE,
                                      verbose=0)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    print('baseline_results')
    image_batch_train, train_labels = next(iter(train_ds))
    image_batch_test, test_labels = next(iter(test_ds))
    print('image_batch_test, test_labels')
    train_predictions = model.predict(image_batch_train, batch_size=BATCH_SIZE)
    test_predictions = model.predict(image_batch_test, batch_size=BATCH_SIZE)
    print('Predictions')
    """### Matriz de Confusión """

    path_file= os.path.join(plots_path, str('Metricas_'+Modelo+'.txt'))
    with open(path_file, 'w') as writefile:
      writefile.write('Metricas obtenidas en primer entrenamiento:\n')
      writefile.write('\n')
      for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
        writefile.write(f'{name} : {value}\n')
      print()

    plot_cm(test_labels, test_predictions,  plots_path, '2')
    print('Plot CM')
    """### ROC"""

    fig = plt.figure()
    plot_roc("Train", train_labels, train_predictions, color=colors[0])
    plot_roc("Test", test_labels, test_predictions, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    i= Metodologia.split(' ')[-1]
    plt.title(f'Curva ROC Modelo {Modelo}, Metodología {i}')
    plt.show()
    fig.savefig(f'{plots_path}/roc_2_{Modelo}.png')
    print('Plot ROC')
    """### PRC"""

    fig = plt.figure()
    plot_prc("Train", train_labels, train_predictions, color=colors[0])
    plot_prc("Test", test_labels, test_predictions, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.title(f'Curva PRC Modelo {Modelo}, Metodología {i}')
    plt.show()
    fig.savefig(f'{plots_path}/prc_2_{Modelo}.png')
    print('Plot PRC')
    print('Fin')

