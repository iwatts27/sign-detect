# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:24:41 2017

@author: iwatts
"""

import numpy as np
import pickle
import time
from keras.models      import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers  import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical


def normalize(im):
    return (np.float64(im) - 127)/127


def shuffle_unison(a, b):
    # From https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


t = time.time()

feat = pickle.load(open("../trainFeat.pkl", "rb"))
feat = normalize(feat)
label = pickle.load(open("../trainLabel.pkl", "rb"))

label_cat = to_categorical(label, int(np.max(label)+1))

shuffle_unison(feat, labelCat)

model = Sequential()
# Modified from https://chsasank.github.io/keras-tutorial.html
model.add(Convolution2D(32, (5, 5), input_shape=feat[0].shape, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))   
     
model.add(Convolution2D(64, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))   
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(43))
model.add(Activation('softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(feat, label_cat, epochs=30, validation_split=0.2, batch_size=4,
                 callbacks=[ModelCheckpoint('model2.h5', save_best_only=True)])

model.save("classifier3.h5")

t2 = time.time()
print('Run time (s):', round(t2-t, 2))