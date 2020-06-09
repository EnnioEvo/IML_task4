import itertools
import os
import pathlib
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from PIL import Image
from math import ceil, floor
from timeit import default_timer as timer
import glob

seed = 470
np.random.seed(seed)
tf.random.set_seed(seed)

shuffle = True
AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("Availables GPU:")
print(tf.config.list_physical_devices('GPU') if tf.config.list_physical_devices('GPU') != [] else 'No GPU available')
#os.environ["TFHUB_CACHE_DIR"] = "C:/Users/Ennio/AppData/Local/Temp/model"

# read features
N_features = 1001
# paths = glob.glob("../data/features_inception_resnet[0-9]*")
# degs = sorted([int(path[33:-4]) for path in paths])
degs = [0, 45, 90, 135, 180, 225, 270, 315, 335]

features_aug = np.zeros([len(degs), 10000, N_features])
for i in range(len(degs)):
    features_aug[i, :, :] = np.array(pd.read_csv(
        '../data/features_inception_resnet' + str(degs[i]) + '.zip', compression='zip', delimiter=',', header=None
    ))

# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', names=['A', 'B', 'C'])
test_triplets_df = pd.read_csv('../data/test_triplets.txt', delimiter=' ', names=['A', 'B', 'C'])
val_triplets_df = pd.read_csv('../data/val_triplets.txt', delimiter=' ', names=['A', 'B', 'C'])

# swap half
N_train_tot = len(train_triplets_df.index)
N_test = len(test_triplets_df.index)
swapped_train_triplets_df = train_triplets_df.iloc[:int(N_train_tot / 2), :]
swapped_train_triplets_df.columns = ['A', 'C', 'B']
train_triplets_df = pd.concat((swapped_train_triplets_df, train_triplets_df.iloc[int(N_train_tot / 2):, :]), sort=True)

# create Y
Y_train_np = np.zeros((N_train_tot, 2))
Y_train_np[:, 0] = (np.arange(N_train_tot) >= int(N_train_tot / 2)) * 1
Y_train_np[:, 1] = 1 - Y_train_np[:, 0]
Y_train_df = pd.DataFrame(Y_train_np, index=range(Y_train_np.shape[0]))

# create val
submit = True
if not submit:
    new_train_index = [i for i in train_triplets_df.index if i not in val_triplets_df.index]
    train_triplets_df = train_triplets_df.loc[new_train_index, :]
    Y_val_np = Y_train_np[val_triplets_df.index, :]
    Y_train_np = Y_train_np[train_triplets_df.index, :]
    N_train = len(train_triplets_df.index)

if shuffle:
    index_permutation = np.random.permutation(train_triplets_df.index)
    train_triplets_df = train_triplets_df.reindex(index_permutation).set_index(np.arange(0, train_triplets_df.shape[0], 1))
    Y_train_df = Y_train_df.loc[index_permutation, :]
    Y_train_np = np.array(Y_train_df)


# define generators
train_permutation = np.random.permutation(range(train_triplets_df.shape[0]))
def X_train_generator():
    train_triplets_loc = np.array(train_triplets_df)
    while True:
        for i in range(len(degs)):
            for row in train_triplets_loc:
                i = np.random.randint(len(degs))
                yield features_aug[i, row[0], :], features_aug[i, row[1], :], features_aug[i, row[2], :]
            train_triplets_loc = train_triplets_loc[train_permutation,:]


def Y_train_generator():
    Y_train_loc = Y_train_np
    while True:
        for y in Y_train_loc:
            y = tf.cast(tf.constant(y), tf.int32, name=None)
            yield (y,)
        Y_train_loc = Y_train_loc[train_permutation, :]


def X_val_generator():
    for _, row in val_triplets_df.iterrows():
        yield features_aug[0, row['A'], :], features_aug[0, row['B'], :], features_aug[0, row['C'], :]


def Y_val_generator():
    for y in Y_val_np:
        y = tf.cast(tf.constant(y), tf.int32, name=None)
        yield (y,)


def X_test_generator(f):
    for _, row in test_triplets_df.iterrows():
        yield features_aug[f, row['A'], :], features_aug[f, row['B'], :], features_aug[f, row['C'], :]

#build datasets
BATCH_SIZE = 64
input_shape = (N_features,)
X_train = tf.data.Dataset.from_generator(X_train_generator,
                                         (tf.float32, tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape(input_shape),) * 3
                                         )
Y_train = tf.data.Dataset.from_generator(Y_train_generator,
                                         (tf.int32,),
                                         output_shapes=(tf.TensorShape((2,)),)
                                         )
X_val = tf.data.Dataset.from_generator(X_val_generator,
                                         (tf.float32, tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape(input_shape),) * 3
                                         )
Y_val = tf.data.Dataset.from_generator(Y_val_generator,
                                         (tf.int32,),
                                         output_shapes=(tf.TensorShape((2,)),)
                                         )


# Y_train = tf.data.Dataset.from_tensor_slices(Y_train_ts)
zipped_train = tf.data.Dataset.zip((X_train, Y_train)).batch(BATCH_SIZE)
zipped_val = tf.data.Dataset.zip((X_val, Y_val)).batch(BATCH_SIZE)


# parameters
neurons = 10
steps_per_epoch = 930 if submit else 910
epochs = 10
optimizer = tf.keras.optimizers.Adam()

# build the model
input_A = tf.keras.layers.Input(shape=input_shape, name='input_A'),
input_B = tf.keras.layers.Input(shape=input_shape, name='input_B'),
input_C = tf.keras.layers.Input(shape=input_shape, name='input_C'),

inputs_AB = [input_A[0], input_B[0]]
inputs_AC = [input_A[0], input_C[0]]

x_AB = tf.keras.layers.Concatenate(axis=1)(inputs_AB)
x_AC = tf.keras.layers.Concatenate(axis=1)(inputs_AC)
x_AB = tf.keras.layers.Dense(neurons)(x_AB)
x_AC = tf.keras.layers.Dense(neurons)(x_AC)
x_AB = tf.keras.layers.BatchNormalization()(x_AB)
x_AC = tf.keras.layers.BatchNormalization()(x_AC)
x_AB = tf.keras.layers.ReLU()(x_AB)
x_AC = tf.keras.layers.ReLU()(x_AC)

x = tf.keras.layers.Concatenate(axis=1)([x_AB, x_AC])
x = tf.keras.layers.Dense(2)(x)
x = tf.keras.layers.BatchNormalization()(x)
output = tf.keras.layers.Softmax()(x)

model = tf.keras.Model(inputs=[input_A, input_B, input_C], outputs=output, name='task3_model')

model.summary()
# tf.keras.utils.plot_model(
#   model, to_file='model_features.png', show_shapes=False, show_layer_names=True)

# compile
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy']
              )


# callbacks
class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)
        print('\nTime elapsed:' + str(timer() - self.starttime))


cb = TimingCallback()
es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', verbose=1, restore_best_weights=True,
                                     patience=3)

# fit
print('Training started')
model.fit(zipped_train, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
          use_multiprocessing=True, workers=1,
          callbacks=[cb, es] if not submit else [cb],
          validation_data=zipped_val if not submit else None)


# debug only
start = timer()
model.predict([np.ones([BATCH_SIZE, N_features]), ] * 3)
end = timer()
elapsed = end - start
print(str(round(elapsed, 2)) + " sec to predict a batch of " + str(BATCH_SIZE)
      + ", 59516 samples will be evaluated in " + str(round(59516 / BATCH_SIZE * elapsed, 2)) + "sec")

#build tests from predict
def X_test_from_feature(f):
    X_test = tf.data.Dataset.from_generator(X_test_generator,
                                            (tf.float32, tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape(input_shape),) * 3,
                                            args=(f,)
                                            ).batch(BATCH_SIZE)
    return X_test

# predict
def batch_predict(X, N):
    X_it = X.as_numpy_iterator()
    Y_batch = np.zeros([0, 2])
    for n in range(0, N, BATCH_SIZE):  # N = 59516 ==>
        start = timer()
        Y_batch = np.row_stack([Y_batch, model.predict(next(X_it))])
        end = timer()
        if n%(20*64) == 0:
            print('Predicted until ' + str(n) + ', ' + str(round(end - start, 2)) + 's')
        end
    print('Predicted')
    return Y_batch

Y_test = batch_predict(X_test_from_feature(0), N_test)
#pd.DataFrame(data=(Y_test[:, 0]), columns=None, index=None).to_csv("../data/submission_float.csv", index=None,
#                                                                   header=None, float_format='%.2f')
pd.DataFrame(data=(Y_test[:, 0] > 0.5) * 1, columns=None, index=None).to_csv("../data/submission.csv", index=None,
                                                                             header=None)
print('Done')
