import sys
import numpy as np
import keras


def Nvidia_CNN(shape):
    model = keras.models.Sequential()
    model.add((keras.layers.BatchNormalization(epsilon=0.001, axis=1,input_shape=shape)))

    model.add(keras.layers.Conv2D(filters=12, kernel_size=(5,5), strides=(1,1), activation='elu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(keras.layers.Conv2D(filters=24, kernel_size=(5,5), activation='elu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(keras.layers.Conv2D(filters=36, kernel_size=(3,3), activation='elu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(keras.layers.Conv2D(filters=48, kernel_size=(3,3), activation='elu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1164, activation='elu'))
    #model.add(keras.layers.Dropout(rate=0.5))

    model.add(keras.layers.Dense(100, activation='elu'))
    #model.add(keras.layers.Dropout(rate=0.5))

    model.add(keras.layers.Dense(50, activation='elu'))
    model.add(keras.layers.Dense(10, activation='elu'))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss='mse')
    model.summary()
    return model

