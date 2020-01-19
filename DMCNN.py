import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda, Input
from keras.backend import resize_images
from keras.layers.normalization import BatchNormalization
import numpy as np
from os.path import join, isfile, isdir, exists
from os import listdir, mkdir, remove
from shutil import copy2
from PIL import Image
from skimage.io import imread
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import arange
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras import optimizers
# Importing necessary functions
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from os import listdir, mkdir
from os.path import isfile, join, isdir, exists
from skimage.transform import resize
from numpy import reshape
from Adam_lr_mult import *
from cv2 import imread, resize
import tensorflow as tf
import cv2



def show_image(img, title):
    plt.title(title)
    plt.imshow(img)
    plt.show()

def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

def add_channels(img):
    img[:, :, 1] = 0
    img[:, :, 2] = 0
    return img


x = []
y = []


mypath = "filtered_dataset/training"
origin_mypath = "origin_filtered_dataset/training"
evalpath = "filtered_dataset/validation"
origin_evalpath = "origin_filtered_dataset/validation"

best_model = ModelCheckpoint(filepath='DMCNN.h5', monitor='val_loss', mode='min', save_best_only=True)


onlyfiles = [f for f in listdir(join(mypath)) if isfile(join(mypath, f))]
for f in onlyfiles:
    im = cv2.cvtColor(imread(join(mypath, f)), cv2.COLOR_BGR2RGB)
    #show_image(im, "original_im")
    im = add_channels(im)
    #show_image(im, "added_channels")
    x.append(im)
    original_im = cv2.cvtColor(imread(join(origin_mypath, f)), cv2.COLOR_BGR2RGB)
    #show_image(original_im, "original_patch")
    y.append(original_im)



x = np.array(x)
y = np.array(y)
randomize = np.arange(len(x))
np.random.shuffle(randomize)
x = x[randomize]
y = y[randomize]

x_eval = []
y_eval = []

onlyfiles = [f for f in listdir(join(evalpath)) if isfile(join(evalpath, f))]
for f in onlyfiles:
    im = cv2.cvtColor(imread(join(evalpath, f)), cv2.COLOR_BGR2RGB)
    #show_image(im, "original_im")
    im = add_channels(im)
    #show_image(im, "added_channels")
    x_eval.append(im)
    original_im = cv2.cvtColor(imread(join(origin_evalpath, f)), cv2.COLOR_BGR2RGB)
    #show_image(original_im, "original_patch")
    y_eval.append(original_im)

x_eval = np.array(x_eval)
y_eval = np.array(y_eval)
randomize = np.arange(len(x_eval))
np.random.shuffle(randomize)
x_eval = x_eval[randomize]
y_eval = y_eval[randomize]


model = Sequential()

# Feature extraction layer
model.add(Conv2D(name='feature_extraction', filters=128, input_shape=(33, 33, 3), kernel_size=(9, 9), use_bias=True, activation='relu'))

# Non-linear mapping layer
model.add(Conv2D(name='mapping', filters=64, kernel_size=(1, 1), use_bias=True, activation='relu'))

# Reconstruction layer
model.add(Conv2D(name='reconstruction', filters=3, kernel_size=(5, 5), use_bias=True))


model.add(UpSampling2DBilinear((33, 33)))

model.summary()
#learning_rate_multipliers = {}
#learning_rate_multipliers['feature_extraction'] = 0.01
#learning_rate_multipliers['mapping'] = 0.01
#learning_rate_multipliers['reconstruction'] = 0.01

#adam_with_lr_multipliers = Adam_lr_mult(multipliers=learning_rate_multipliers)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mean_squared_error')

training = model.fit(x, y, batch_size=32, epochs=200, validation_split=0.1, callbacks=[best_model])

history = training.history

# Plot the training loss
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

# Show the figure
plt.show()

y_predict = model.predict(x_eval)

for i in range(len(y_predict)):
    show_image(Image.fromarray(y_predict[i], 'RGB'), "prediction")
    show_image(y_eval[i], "reality")





