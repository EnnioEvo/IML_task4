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

np.random.seed(400)
shuffle = True
AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("Availables GPU:")
print(tf.config.list_physical_devices('GPU') if tf.config.list_physical_devices('GPU') != [] else 'No GPU available')
os.environ["TFHUB_CACHE_DIR"] = "C:/Users/Ennio/AppData/Local/Temp/model"

# read features
features = np.array(pd.read_csv('../data/features_resnet_from_best_code.csv', delimiter=',', header=None))
BATCH_SIZE = 64
# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header=None)
test_triplets_df = pd.read_csv('../data/test_triplets.txt', delimiter=' ', header=None)
train_triplets_df.columns = ['A', 'B', 'C']
test_triplets_df.columns = ['A', 'B', 'C']
N_train = len(train_triplets_df.index)
N_test = len(test_triplets_df.index)

# swap half
N_train = len(train_triplets_df.index)
N_test = len(test_triplets_df.index)
swapped_train_triplets_df = train_triplets_df.iloc[:int(N_train / 2), :]
swapped_train_triplets_df.columns = ['A', 'C', 'B']
train_triplets_df = pd.concat((swapped_train_triplets_df, train_triplets_df.iloc[int(N_train / 2):, :]), sort=True)
# train_triplets_dict = {index: list(row) for index, row in train_triplets_df.iterrows()}

# create Y
Y_train_np = np.zeros((N_train,2))
Y_train_np[:,0] = (np.arange(N_train) >= int(N_train / 2)) * 1
Y_train_np[:,1] = 1 - Y_train_np[:,0]

if shuffle:
    rd_permutation = np.random.permutation(train_triplets_df.index)
    train_triplets_df = train_triplets_df.reindex(rd_permutation).set_index(np.arange(0, train_triplets_df.shape[0], 1))
    Y_train_np = Y_train_np[rd_permutation,:]
Y_train_ts = tf.constant(Y_train_np)

# build test and train
print()
def X_train_generator():
    train_triplets_loc = train_triplets_df
    while True:
        for _, row in train_triplets_loc.iterrows():
            yield features[row['A'], :], features[row['B'], :], features[row['C'], :]
        train_triplets_loc = train_triplets_loc.reindex(rd_permutation).set_index(np.arange(0, train_triplets_loc.shape[0], 1))


def Y_train_generator():
    Y_train_loc = Y_train_np
    while True:
        for y in Y_train_loc:
            y = tf.cast(tf.constant(y), tf.int32, name=None)
            yield (y,)
        Y_train_loc = Y_train_loc[rd_permutation,:]


def X_test_generator():
    for _, row in test_triplets_df.iterrows():
        yield features[row['A'], :], features[row['B'], :], features[row['C'], :]


input_shape = (2048,)
X_train = tf.data.Dataset.from_generator(X_train_generator,
                                         (tf.float32, tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape(input_shape),) * 3
                                         )
Y_train = tf.data.Dataset.from_generator(Y_train_generator,
                                         (tf.int32,),
                                         output_shapes=(tf.TensorShape((2,)),)
                                         )
X_test = tf.data.Dataset.from_generator(X_test_generator,
                                        (tf.float32, tf.float32, tf.float32),
                                        output_shapes=(tf.TensorShape(input_shape),) * 3,
                                        ).batch(BATCH_SIZE)

#Y_train = tf.data.Dataset.from_tensor_slices(Y_train_ts)
zipped_train = tf.data.Dataset.zip((X_train, Y_train)).batch(BATCH_SIZE)

# parameters
neurons = 30
steps_per_epoch = 930
epochs = 5
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# build the model
input_A = tf.keras.layers.Input(shape=input_shape, name='input_A'),
input_B = tf.keras.layers.Input(shape=input_shape, name='input_B'),
input_C = tf.keras.layers.Input(shape=input_shape, name='input_C'),

inputs_AB = [input_A[0], input_B[0]]
inputs_AC = [input_A[0], input_C[0]]

x_AB = tf.keras.layers.Concatenate(axis=1)(inputs_AB)
x_AC = tf.keras.layers.Concatenate(axis=1)(inputs_AC)
x_AB = tf.keras.layers.Dense(neurons, activation='relu')(x_AB)
x_AC = tf.keras.layers.Dense(neurons, activation='relu')(x_AC)

# x_AB = tf.keras.layers.Subtract()(inputs_AB)
# x_AC = tf.keras.layers.Subtract()(inputs_AC)

x = tf.keras.layers.Concatenate(axis=1)([x_AB, x_AC])
output = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=[input_A, input_B, input_C], outputs=output, name='task3_model')

model.summary()
tf.keras.utils.plot_model(
   model, to_file='model_features.png', show_shapes=False, show_layer_names=True)

#compile
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy']
              )
#fit
print('Training started')
model.fit(zipped_train, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, use_multiprocessing=True)

#debug only
start = timer()
model.predict( [np.ones([BATCH_SIZE,2048]),]*3 )
end = timer()
elapsed = end - start
print(str(round(elapsed,2)) + " sec to predict a batch of " + str(BATCH_SIZE)
      + ", 59516 samples will be evaluated in " + str(round(59516/BATCH_SIZE*elapsed, 2)) + "sec")

#predict
def batch_predict(X, N):
  X_it = X.as_numpy_iterator()
  Y_batch = np.zeros([0,2])
  for n in range(0, N, BATCH_SIZE): #N = 59516 ==>
      start = timer()
      Y_batch = np.row_stack([Y_batch, model.predict(next(X_it))])
      end = timer()
      print('Predicted until ' + str(n) + ', ' + str(round(end-start,2)) + 's')
  print('Predicted')
  return Y_batch

Y_test = batch_predict(X_test, N_test)
pd.DataFrame(data=(Y_test[:,0]), columns=None, index=None).to_csv("../data/sumbission_float.csv", index=None, header=None, float_format='%.2f')
pd.DataFrame(data=(Y_test[:,0]>0.5)*1, columns=None, index=None).to_csv("../data/sumbission.csv", index=None, header=None)
print('Done')
