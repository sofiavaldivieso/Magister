import tensorflow as tf
from tensorflow.keras import regularizers


def modelo(IMG_SIZE, model_name, metrics):
    data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                             tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                             ])
    IMG_SHAPE = IMG_SIZE + (3,)
    if model_name == 'ResNet50':
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.applications.ResNet50(include_top=False,
                                                    weights='imagenet',
                                                    input_tensor=None,
                                                    input_shape=IMG_SHAPE)
    elif model_name == 'Xception':
        preprocess_input = tf.keras.applications.xception.preprocess_input
        base_model = tf.keras.applications.Xception(include_top=False,
                                                    weights='imagenet',
                                                    input_tensor=None,
                                                    input_shape=IMG_SHAPE)
    elif model_name == 'MobileNetV2':
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                       weights='imagenet',
                                                       input_tensor=None,
                                                       input_shape=IMG_SHAPE)
    elif model_name == 'Inception':
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                       weights='imagenet',
                                                       input_tensor=None,
                                                       input_shape=IMG_SHAPE)                                                  
    global_avg_pooling=tf.keras.layers.GlobalAveragePooling2D()
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_avg_pooling(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    model.summary()
    return model, base_model



def late_fusion(IMG_SIZE, model_name, metrics, pesos1 = None, pesos2=None):
    model1, base_model1 = modelo(IMG_SIZE, model_name, metrics)
    model2, base_model2 = modelo(IMG_SIZE, model_name, metrics)
    if pesos1 is not None:
      model1.load_weights(pesos1)
    if pesos2 is not None:
      model2.load_weights(pesos2)
      
    model1 = tf.keras.Model(inputs=model1.input, outputs=model1.layers[-3].output)
    model2 = tf.keras.Model(inputs=model2.input, outputs=model2.layers[-3].output)

    for layer in model2.layers:
      layer._name = layer.name.split('_')[0]+str('_FUNDUS')
    for layer in model1.layers:
      layer._name = layer.name.split('_')[0]+str('_OCT')

    crop_OCT = tf.keras.layers.Cropping2D(cropping=((0, 0), (495, 0))) #OCT
    crop_FUNDUS = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 769))) #FUNDUS
    resize_OCT = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE[0], IMG_SIZE[1])])
    resize_FUNDUS = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE[0], IMG_SIZE[1])])

    input_OCT= tf.keras.Input(shape=(496, 1264, 3))

    input1= crop_OCT(input_OCT)
    input1= resize_OCT(input1)
    input1 = tf.keras.Model(inputs=input_OCT, outputs=input1)

        
    input2= crop_FUNDUS(input_OCT)
    input2= resize_FUNDUS(input2)
    input2 = tf.keras.Model(inputs=input_OCT, outputs=input2)

    x1 = model1(input1.output) 
    x2 = model2(input2.output)
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_OCT, outputs=outputs)
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    model.summary()
    return model, base_model1, base_model2
