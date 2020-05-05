# -*- coding: utf-8 -*-
#https://colab.research.google.com/drive/1tUFcYSttIgbCQ0atWOdcLsKtEB5hT8p-

import itertools
import os
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from math import ceil, floor

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
print(tf.config.list_physical_devices('GPU'))

#read triplets
train_triplets = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header = None)
train_triplets.columns = ['A', 'B', 'C']

#swap half
N_train = len(train_triplets.index)
swapped_train_triplets = train_triplets.iloc[:int(N_train/2),:]
swapped_train_triplets.columns = ['A', 'C', 'B']
new_train_triplets = pd.concat((swapped_train_triplets, train_triplets.iloc[int(N_train/2)+1:,:]), sort=True)

#create Y
Y_train = (np.arange(N_train)<int(N_train/2))*1

module_selection = ("inception_v3", 299) 
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32
data_dir = '../data/food/'

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = False 
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)

do_fine_tuning = False

print("Building model with", MODULE_HANDLE)

model_A = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning, name='layer_A'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(2,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model_B = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning, name='layer_B'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(2,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

model_C = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning, name='layer_C'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(2,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model_A.build((None,)+IMAGE_SIZE+(3,))
model_B.build((None,)+IMAGE_SIZE+(3,))
model_C.build((None,)+IMAGE_SIZE+(3,))

outputs_AB = [model_A.output, model_B.output]
outputs_AC = [model_A.output, model_C.output]

x_AB = tf.keras.layers.Concatenate(axis=1)(outputs_AB)
x_AC = tf.keras.layers.Concatenate(axis=1)(outputs_AC)

x_AB = tf.keras.layers.Dense(1000, activation='relu')(x_AB)
x_AC = tf.keras.layers.Dense(1000, activation='relu')(x_AC)

x = tf.keras.layers.Concatenate(axis=1)([x_AB, x_AC])
output = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=[model_A.input, model_B.input, model_C.input], outputs=output, name='task3_model')

#model.summary()

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True)

model.compile(optimizer=tf.keras.optimizers.Adadelta(),
              loss=tf.keras.losses.mean_squared_error,
              metrics=[tf.keras.metrics.categorical_accuracy]
              )
model.fit(train_generator, Y_train, verbose=1)


# print("Building model with", MODULE_HANDLE)
# model = tf.keras.Sequential([
#     # Explicitly define the input shape so the model can be properly
#     # loaded by the TFLiteConverter
#     tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
#     hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning, name='ciao'),
#     tf.keras.layers.Dropout(rate=0.2),
#     tf.keras.layers.Dense(2,
#                           kernel_regularizer=tf.keras.regularizers.l2(0.0001))
# ])
# model.build((None,)+IMAGE_SIZE+(3,))
# model.summary()
# model.input
