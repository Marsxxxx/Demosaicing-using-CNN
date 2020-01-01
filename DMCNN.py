from keras import Sequential
from keras.layers import Conv2D
import numpy as py
import keras
from Adam_lr_mult import *

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
