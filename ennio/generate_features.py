import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from PIL import Image
from skimage.util import random_noise
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

seed = 470
np.random.seed(seed)
tf.random.set_seed(seed)


shuffle = True
AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("Availables GPU:")
#print([dev.physical_device_desc for dev in tf.python.client.device_lib.list_local_devices()])
#print('Gpu:' + tf.python.client.device_lib.list_local_devices()[-1].physical_device_desc)
# os.environ["TFHUB_CACHE_DIR"] = "C:/Users/Ennio/AppData/Local/Temp/model"

# read triplets
train_triplets_df = pd.read_csv('../data/train_triplets.txt', delimiter=' ', header=None)
test_triplets_df = pd.read_csv('../data/test_triplets.txt', delimiter=' ', header=None)
train_triplets_df.columns = ['A', 'B', 'C']
test_triplets_df.columns = ['A', 'B', 'C']
N_train = len(train_triplets_df.index)
N_test = len(test_triplets_df.index)

pixels = 299
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4"
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

# build the model draft1
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=True, name='layer_A'),
])
model.build((None,) + IMAGE_SIZE + (3,))
model.summary()


# read images
def label2path(label):
    return '../data/food/' + str(label).zfill(5) + '.jpg'

# predict
BATCH_SIZE = 1000
for deg in [0, 45, 90, 135, 180, 225, 270, 315, 335]:
    features = np.zeros([10000, 1001])
    for b in range(int(10000 / BATCH_SIZE)):
        batch_images = np.zeros([BATCH_SIZE, pixels, pixels, 3])
        for label in range(BATCH_SIZE):
            image = np.array(Image.open(label2path(b * BATCH_SIZE + label)).resize((pixels, pixels), Image.BICUBIC))
            batch_images[label, :, :, :] = Image.fromarray(image).rotate(deg)
            if label % 100 == 0:
                print(b * BATCH_SIZE + label)
        features[b * BATCH_SIZE:(b + 1) * BATCH_SIZE, :] = model.predict(batch_images / 255)
    pd.DataFrame(data=features, columns=None, index=None).to_csv(
        "../data/features_inception_resnet" + str(deg) + ".zip", compression='zip', index=None, header=None,
        float_format='%.8f')
print("Features generated")
