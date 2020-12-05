import sys
import numpy as np
import keras


def Nvidia_CNN(chan,h,w):
    model = keras.models.Sequential()
    model.add((keras.layers.BatchNormalization(epsilon=0.001, axis=1,input_shape=(h, w, chan))))
    model.add(keras.layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='elu'))
    model.add(keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='elu'))
    model.add(keras.layers.Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='elu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='elu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.8))
    model.add(keras.layers.Dense(units=100, activation='elu'))
    model.add(keras.layers.Dropout(rate=0.8))
    model.add(keras.layers.Dense(units=50, activation='elu'))
    model.add(keras.layers.Dropout(rate=0.7))    
    model.add(keras.layers.Dense(units=10, activation='elu'))
    model.add(keras.layers.Dense(units=1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),loss='mse')
    model.summary()
    return model
    
