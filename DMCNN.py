import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
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
from skimage.io import imread, imsave
from numpy import reshape
from Adam_lr_mult import *
import cv2


x = []
y = []


mypath = "filtered_dataset/training"
evalpath = "filtered_dataset/validation"

dirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]


for dir in dirs:
    onlyfiles = [f for f in listdir(join(mypath, dir)) if isfile(join(mypath, dir, f))]
    for f in onlyfiles:
        x.append(imread(join(mypath, dir, f)))
        if dir == "glaucomatous":
            y.append([0])
        else:
            y.append([1])


x = np.array(x)
y = np.array(y)
y = to_categorical(y, 2)
randomize = np.arange(len(x))
np.random.shuffle(randomize)
x = x[randomize]
y = y[randomize]

x_eval = []
y_eval = []

for dir in dirs:
    onlyfiles = [f for f in listdir(join(evalpath, dir)) if isfile(join(evalpath, dir, f))]
    for f in onlyfiles:
        x_eval.append(imread(join(evalpath, dir, f)))
        if dir == "glaucomatous":
            y_eval.append([0])
        else:
            y_eval.append([1])

x_eval = np.array(x_eval)
y_eval = np.array(y_eval)
y_eval = to_categorical(y_eval, 2)
randomize = np.arange(len(x_eval))
np.random.shuffle(randomize)
x_eval = x_eval[randomize]
y_eval = y_eval[randomize]


model = Sequential()

# Feature extraction layer
model.add(Conv2D(name='feature_extraction', filters=128, input_shape=(33, 33, 3), kernel_size=(9, 9), use_bias=True,
                 activation='relu'))

# Non-linear mapping layer
model.add(Conv2D(name='mapping', filters=64, kernel_size=(1, 1), use_bias=True, activation='relu'))

# Reconstruction layer
model.add(Conv2D(name='reconstruction', filters=3, kernel_size=(5, 5), use_bias=True))

model.summary()

learning_rate_multipliers = {}
learning_rate_multipliers['feature_extraction'] = 1
learning_rate_multipliers['mapping'] = 1
learning_rate_multipliers['reconstruction'] = 0.1

adam_with_lr_multipliers = Adam_lr_mult(multipliers=learning_rate_multipliers)

model.compile(optimizer=adam_with_lr_multipliers, loss='mean_squared_error')

training = model.fit(x, y, batch_size=)


