# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:23:13 2017
Updated on Nov 14 2017
@author: Zain
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Permute
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.layers import RandomTranslation
from layers import RandomTimeMasking, RandomFrequencyMasking

def CRNN2D(X_shape, nb_classes):
    '''
    Model used for evaluation in paper. Inspired by K. Choi model in:
    https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py
    '''

    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    #Data Augmentation Variables for LB Policy
    W = 80
    F = 27
    mF = 1
    T = 100
    p = 1
    mT = 1
    num_time_channels = 911
    num_freq_channels = 128
    

    # Create sequential model and normalize along frequency axis
    model = Sequential()

    # Add data augmentation
    model.add(RandomTranslation((0,0),((0.0878),(0.9122)), fill_mode='wrap', input_shape=input_shape))
    model.add(RandomFrequencyMasking(F, num_freq_channels, input_shape=input_shape))
    model.add(RandomTimeMasking(T, num_time_channels, input_shape=input_shape))

    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0]))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


if __name__ == '__main__':
    pass