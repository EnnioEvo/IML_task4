import itertools
import os
import pathlib
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from math import ceil, floor

AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("Availebles GPU:")
print(tf.config.list_physical_devices('GPU') if tf.config.list_physical_devices('GPU') != [] else 'No GPU available')

# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header=None)
train_triplets_df.columns = ['A', 'B', 'C']

# swap half
N_train = len(train_triplets_df.index)
swapped_train_triplets_df = train_triplets_df.iloc[:int(N_train / 2), :]
swapped_train_triplets_df.columns = ['A', 'C', 'B']
train_triplets_df = pd.concat((swapped_train_triplets_df, train_triplets_df.iloc[int(N_train / 2) + 1:, :]), sort=True)
# train_triplets_dict = {index: list(row) for index, row in train_triplets_df.iterrows()}

# create Y
Y_train = (np.arange(N_train) < int(N_train / 2)) * 1

module_selection = ("inception_v3", 299)
handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32
data_dir = '../data/food/'
data_dir = pathlib.Path(data_dir)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])


def label2path(label):
    return '..\\data\\food\\' + str(label).zfill(5) + '.jpg'


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-1] == CLASS_NAMES


def get_img(file_path):
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, IMAGE_SIZE)


def build_image_triplet(label_triple):
    #return [get_img(label2path(label_triple[i])) for i in range(3)]
    return get_img(label2path(label_triple[0])), get_img(label2path(label_triple[1])), get_img(label2path(label_triple[1]))

#list_ds = tf.data.Dataset.list_files(str(data_dir / '*'), shuffle=False)
#images_ds = list_ds.map(get_img, num_parallel_calls=AUTOTUNE)
# X_train = range_ds.map(lambda i: build_image_triplet(tf.gather(train_triplets_dict,i)), num_parallel_calls=AUTOTUNE)
# train_generator = (build_image_triplet([row.A,row.B,row.C]) for index, row in train_triplets_df.iterrows())

def train_generator():
    for index, row in train_triplets_df.iterrows():
        yield build_image_triplet(list(row))


X_train = tf.data.Dataset.from_generator(train_generator,
                                         (tf.float32, tf.float32, tf.float32),
                                         output_shapes=(tf.TensorShape([pixels, pixels, 3]),
                                                        tf.TensorShape([pixels, pixels, 3]),
                                                        tf.TensorShape([pixels, pixels, 3]))
                                         )

# build the model
print("Building model with", MODULE_HANDLE)
do_fine_tuning = False
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
model_A.build((None,) + IMAGE_SIZE + (3,))
model_B.build((None,) + IMAGE_SIZE + (3,))
model_C.build((None,) + IMAGE_SIZE + (3,))

outputs_AB = [model_A.output, model_B.output]
outputs_AC = [model_A.output, model_C.output]

x_AB = tf.keras.layers.Concatenate(axis=1)(outputs_AB)
x_AC = tf.keras.layers.Concatenate(axis=1)(outputs_AC)

x_AB = tf.keras.layers.Dense(1000, activation='relu')(x_AB)
x_AC = tf.keras.layers.Dense(1000, activation='relu')(x_AC)

x = tf.keras.layers.Concatenate(axis=1)([x_AB, x_AC])
output = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs=[model_A.input, model_B.input, model_C.input], outputs=output, name='task4_model')
# model.summary()

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
