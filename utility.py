import os
import dill
import random
import itertools
import logging
import pickle

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from scipy import stats

def load_dataset(song_folder_name='dataset', producer_folder='music'):
    """This function loads the dataset based on a location;
     it returns a list of spectrograms
     and their corresponding artists/song names"""
    
    # get list of artists to verify artist names later
    artists = os.listdir(producer_folder)

    # Get all songs saved as numpy arrays in the given folder
    song_list_path = os.path.join(os.getcwd(), song_folder_name)
    song_list = os.listdir(song_list_path)

    # Create empty lists
    artist = []
    spectrogram = []

    # Load each song into memory if the artist is included in list and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        if loaded_song[0] in artists:
            artist.append(loaded_song[0])
            spectrogram.append(loaded_song[1])

    return artist, spectrogram

def load_dataset_song_split(random_state:int = 42,
                            song_folder_name='dataset',
                            producer_folder='music',
                            test_split_size=0.2):
    """Splits the dataset into testing and training subsets."""

    artist, spectrogram = load_dataset(song_folder_name=song_folder_name,
                           producer_folder=producer_folder)
    # train and test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        spectrogram, artist, test_size=test_split_size, stratify=artist,
        random_state = random_state)

    return Y_train, X_train, \
           Y_test, X_test

def slice_songs(X, Y, slice_length=911):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []

    # Slice up songs using the length specified
    for i, song in enumerate(X):
        slices = int(song.shape[1] / slice_length)
        for j in range(slices - 1):
            spectrogram.append(song[:, slice_length * j:slice_length * (j + 1)]) # select all rows and length to slice length
            artist.append(Y[i])

    return np.array(spectrogram), np.array(artist) # keras expects numpy arrays

def encode_labels(Y,label_encoder=None, save_le = False):
    """Encodes target variables into numbers and then one hot encodings

    Can also save the label encoder as a pickle object for future use by setting save_le to True. 
    
    """

    # Encode the labels
    if label_encoder is None:
        label_encoder = preprocessing.LabelEncoder()
        Y_integer_encoded = label_encoder.fit_transform(Y)

        if save_le:
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f) # save encoder for decoding in the future

    else:
        Y_integer_encoded = label_encoder.transform(Y)
        print(Y_integer_encoded)

    # convert into one hot encoding
    Y_enc = keras.utils.to_categorical(Y_integer_encoded)
    print(label_encoder.classes_)

    # return encoders to re-use on other data
    return Y_enc, label_encoder

def plot_confusion_matrix(model, x_test, y_test, le = None):
    '''Creates confusion matrix from test set. Provide label encoder to get human-readable labels.'''

    # Predict
    y_prediction = model.predict(x_test)
    y_prediction = np.argmax (y_prediction, axis = 1)
    y_test=np.argmax(y_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    cm = confusion_matrix(y_test, y_prediction , normalize='pred')

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # get class names from label encoder
    class_names = le.classes_

    # We want to show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    # fmt = 'd'
    print(type(cm[1,1]))
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(np.round(cm[i, j], 3)),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
