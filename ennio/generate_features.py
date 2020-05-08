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
# Save image in set directory
# Read RGB image


np.random.seed(400)
shuffle = True
AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("Availables GPU:")
print(tf.config.list_physical_devices('GPU') if tf.config.list_physical_devices('GPU') != [] else 'No GPU available')
os.environ["TFHUB_CACHE_DIR"] = "C:/Users/Ennio/AppData/Local/Temp/model"

# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header=None)
test_triplets_df = pd.read_csv('../data/test_triplets.txt', delimiter=' ', header=None)
train_triplets_df.columns = ['A', 'B', 'C']
test_triplets_df.columns = ['A', 'B', 'C']
N_train = len(train_triplets_df.index)
N_test = len(test_triplets_df.index)

module_selection = ("inception_v3", 299)
handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

data_dir = '../data/food/'
data_dir = pathlib.Path(data_dir)


# build the model draft1
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=False, name='layer_A'),
    #tf.keras.layers.Dropout(rate=0.2),
    #tf.keras.layers.Dense(2,
    #                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

#read images
def label2path(label):
    return '../data/food/' + str(label).zfill(5) + '.jpg'


BATCH_SIZE = 1000
features = np.zeros([10000, 2048])
for b in range(int(10000/BATCH_SIZE)):
    batch_images = np.zeros([BATCH_SIZE, 299, 299, 3])
    for label in range(BATCH_SIZE):
        image = Image.open(label2path(label))
        image = image.resize((299, 299))
        batch_images[label,:,:,:] = image
        if label%10==0:
            print(label)
    features[b*BATCH_SIZE:(b+1)*BATCH_SIZE,:] = model.predict(batch_images)

pd.DataFrame(data=features, columns=None, index=None).to_csv("features.csv", index=None, header=None)


print()




