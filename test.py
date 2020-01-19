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
from numpy import reshape
from Adam_lr_mult import *
from cv2 import imread

x = []
y = []


mypath = "filtered_dataset/training"
origin_mypath = "origin_filtered_dataset/training"
evalpath = "filtered_dataset/validation"
origin_evalpath = "origin_filtered_dataset/validation"

def add_channels(img):
    img[:, :, 1] = 0
    img[:, :, 2] = 0
    return img

onlyfiles = [f for f in listdir(join(mypath)) if isfile(join(mypath, f))]
for f in onlyfiles:
    im = imread(join(mypath, f))
    im = add_channels(im)
    x.append(im)
    y.append(imread(join(origin_mypath, f)))


x = np.array(x)
y = np.array(y)
randomize = np.arange(len(x))
np.random.shuffle(randomize)
x = x[randomize]
y = y[randomize]