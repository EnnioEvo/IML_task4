# -*- coding: utf-8 -*-
"""task4?last.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1953Kgobb7VG4DQdj-sVBuMFALaITPx6Y
"""

import itertools
import os
import pathlib
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from math import ceil, floor

np.random.seed(470)
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)


# =============== MULTI GPU STRATEGY DEFINITION =============== #

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE_PER_REPLICA = 64 #                                                 <=INCREASE ONLY IN GPU  
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

AUTOTUNE = tf.data.experimental.AUTOTUNE

# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header=None)
train_triplets_df.columns = ['A', 'B', 'C']

# swap half
N_train = len(train_triplets_df.index)
swapped_train_triplets_df = train_triplets_df.iloc[:int(N_train / 2), :]
swapped_train_triplets_df.columns = ['A', 'C', 'B']
train_triplets_df = pd.concat((swapped_train_triplets_df, train_triplets_df.iloc[int(N_train / 2):, :]), sort=True)

# create Y
Y_train_np = (np.arange(N_train) >= int(N_train / 2)) * 1

shuffle = True
if shuffle:
    rd_permutation = np.random.permutation(train_triplets_df.index)
    train_triplets_df = train_triplets_df.reindex(rd_permutation).set_index(np.arange(0, train_triplets_df.shape[0], 1))
    Y_train_np = Y_train_np[rd_permutation]

#tensor for Y_train
Y_train_ts = tf.constant(Y_train_np)

handle_base, pixels = ("resnet_v2_50", 224)
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))


def label2path(label):
    return '../data/food/' + str(label).zfill(5) + '.jpg'


def get_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img/255


def build_image_triplet(label_triple):
    return (get_img(label2path(label_triple[0])), get_img(label2path(label_triple[1])),
        get_img(label2path(label_triple[2])))
    

def X_train_generator():
    for index, row in train_triplets_df.iterrows():
        yield build_image_triplet(list(row))


def X_test_generator():
    for index, row in test_triplets_df.iterrows():
        yield build_image_triplet(list(row))

#build X and Y
X_train = tf.data.Dataset.from_generator(X_train_generator,
                                             (tf.float32, tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape([pixels, pixels, 3]),) * 3
                                             )

Y_train_ts = tf.cast(Y_train_ts, tf.int32, name=None)
Y_train = tf.data.Dataset.from_tensor_slices(Y_train_ts)
zipped_train = tf.data.Dataset.zip((X_train, Y_train)).batch(BATCH_SIZE)

# debug only
X_train_it = X_train.as_numpy_iterator()
Y_train_it = Y_train.as_numpy_iterator()
zipped_train_it = zipped_train.as_numpy_iterator()
#next(X_train_it)
#next(Y_train_it)
# next(zipped_train_it)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#imgplot = plt.imshow(get_img(label2path(1)))
#print(get_img(label2path(1)))

do_fine_tuning = False
print("Building model with", MODULE_HANDLE)

model_A = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning, name='layer_A'),
                                ])

model_B = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning, name='layer_B'),
                                ])

model_C = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning, name='layer_C'),
                                ])

model_A.build((None,)+IMAGE_SIZE+(3,))
model_B.build((None,)+IMAGE_SIZE+(3,))
model_C.build((None,)+IMAGE_SIZE+(3,))

outputs_AB = [model_A.output, model_B.output]
outputs_AC = [model_A.output, model_C.output]

x_AB = tf.keras.layers.Concatenate(axis=1)(outputs_AB)
x_AC = tf.keras.layers.Concatenate(axis=1)(outputs_AC)

x_AB = tf.keras.layers.Dropout(rate=0.25)(x_AB)
x_AC = tf.keras.layers.Dropout(rate=0.25)(x_AC)

x_AB = tf.keras.layers.Dense(100, activation='relu')(x_AB)
x_AC = tf.keras.layers.Dense(100, activation='relu')(x_AC)

x = tf.keras.layers.Concatenate(axis=1)([x_AB, x_AC])
x = tf.keras.layers.Dropout(rate=0.2)(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=[model_A.input, model_B.input, model_C.input], outputs=output, name='task4_model')

model.summary()

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy']
              )

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=10)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                                                 save_weights_only=True,
                                                 monitor='accuracy', mode='max',
                                                 verbose=1)

print('Training started')
model.fit(zipped_train, epochs=50, callbacks=[es, cp_callback])

#model = make_model()
model.load_weights(checkpoint_path)

# read triplets
test_triplets_df = pd.read_csv('test_triplets.txt', delimiter=' ', header=None)
test_triplets_df.columns = ['A', 'B', 'C']
N_test = test_triplets_df.shape[0]
handle_base, pixels = ("resnet50", 224)
IMAGE_SIZE = (pixels, pixels)


def label2path(label):
    return '/content/food/' + str(label).zfill(5) + '.jpg'

def get_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img/255


def build_image_triplet(label_triple):
    return (get_img(label2path(label_triple[0])), get_img(label2path(label_triple[1])),
        get_img(label2path(label_triple[2])))
    

def X_train_generator():
    for index, row in test_triplets_df.iterrows():
        yield build_image_triplet(list(row))

#build X and Y
X_train = tf.data.Dataset.from_generator(X_train_generator,
                                             (tf.float32, tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape([pixels, pixels, 3]),) * 3
                                             )

zipped_train = tf.data.Dataset.zip((X_train)).batch(BATCH_SIZE)

Y_pred = np.zeros((1,2))
X_it = zipped_train.as_numpy_iterator()

for batch in range(0, N_test+BATCH_SIZE, BATCH_SIZE):
    Y_pred =  np.vstack((Y_pred, model.predict(next(X_it)) ))
    if batch%640*4==0: print(batch, ' oveer: ',N_test)

Y_submission = Y_pred[1:,0]>=Y_pred[1:,1]

Y_submission.shape

np.savetxt(r'submission.txt', Y_submission, fmt='%d')

