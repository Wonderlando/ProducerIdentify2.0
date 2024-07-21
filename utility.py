import os
import dill
import pickle
import librosa
import logging
from pydub import AudioSegment

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
    song_name = [] # ADD FUNC

    # Load each song into memory if the artist is included in list and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        if loaded_song[0] in artists:
            artist.append(loaded_song[0])
            spectrogram.append(loaded_song[1])
            song_name.append(loaded_song[2])

    return artist, spectrogram, song_name

def load_dataset_song_split(song_folder_name='dataset',
                            producer_folder='music',
                            test_split_size=0.2):
    """Splits the dataset into testing and training subsets.
    
    Always keep the random_state=42 for testing consistency purposes!!"""

    _random_state = 42

    artist, spectrogram, song_name = load_dataset(song_folder_name=song_folder_name,
                           producer_folder=producer_folder)
    # train and test split
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        spectrogram, artist, song_name, test_size=test_split_size, stratify=artist,
        random_state = _random_state)
    
    return X_train, X_test, \
            Y_train, Y_test, \
            S_train, S_test

def slice_songs(X, Y, S, slice_length=911):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []
    song_name = []

    # Slice up songs using the length specified
    for i, song in enumerate(X):
        slices = int(song.shape[1] / slice_length)
        for j in range(slices - 1):
            spectrogram.append(song[:, slice_length * j:slice_length * (j + 1)]) # select all rows and length to slice length
            artist.append(Y[i])
            song_name.append(S[i])

    return np.array(spectrogram), np.array(artist), np.array(song_name) # keras expects numpy arrays

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
    print(y_test.shape)
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

def analyze_misclassificaitons(x_test, y_pred, y_test, s_test, le):
    '''Allows user to visualize misclassifications of model'''
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
    print(misclassified_indices)

    nrows = misclassified_indices.shape[0]
    ncols = 1
    plt.figure(figsize=(nrows, ncols))

    for i in range(misclassified_indices.shape[0]):
        plt.subplot(nrows,ncols, i+1)
        plt.imshow(x_test[misclassified_indices[i]].squeeze(), cmap='viridis')
        song_name = s_test[misclassified_indices[i]]
        true_label = le.classes_[y_true_classes[misclassified_indices[i]]]
        pred_label = le.classes_[y_pred_classes[misclassified_indices[i]]]
        plt.title(f'song name: {song_name}, true label: {true_label}, Predicted label: {pred_label}')
    
    plt.tight_layout()
    plt.show()

def predict_artist(song_path, model_path, label_encoder):

    sr=16000
    n_mels=128
    n_fft=2048
    hop_length=512

    slice_length = 911

    ### Create mel spectrogram and convert it to the log scale
    ## load song
    print('Loading song...')
    try:
        y, sr = librosa.load(song_path, sr=sr)
    except Exception as e:
        logging.error(f"Librosa error processing {song_path}: {e}")
        # Fallback: Use pydub to decode the MP3 file
        try:
            audio = AudioSegment.from_file(song_path)
            y = np.array(audio.get_array_of_samples())
            sr = audio.frame_rate
        except Exception as e:
            logging.error(f"Pydub error processing {song_path}: {e}")

    ## create mel spec
    print('Creating mel-spectrogram...')
    try:
        print(len(y))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
        data = librosa.power_to_db(S, ref=1.0)
        print(data.shape)
    except Exception as e:
        RuntimeError(f"Error processing {song_path}: {e}")

    ## slice song
    print('Slicing spectrograms...')
    spectrograms = [] # stores spectrograms
    num_slices = int(data.shape[1] / slice_length) # number of slices required

    for j in range(num_slices):
        spectrograms.append(data[:, slice_length * j:slice_length * (j + 1)])
    spectrograms = np.array(spectrograms) # convert to numpy array to add channel dimension
    print(spectrograms.shape)
    spectrograms = spectrograms.reshape(spectrograms.shape + (1,)) # add channel dimension
    print('Input shape:', spectrograms.shape)

    ### Call model
    print('Loading model...')
    model = keras.models.load_model(model_path)
    results = model.predict(spectrograms)

    # create dictionary of results
    final_results = {artist:percent for artist, percent in zip(label_encoder.classes_, results[-1])} # last result is all we care about
    print(final_results)

    # visualize results
    fig, ax = plt.subplots()
    print(label_encoder.classes_)
    ax.pie(results[-1], labels = label_encoder.classes_, autopct='%1.1f%%') # again, last result is all we care about

    plt.show()

if __name__ == '__main__':

    _, x_test, _, y_test, _, s_test = load_dataset_song_split()
    x_test, y_test, s_test = slice_songs(x_test, y_test, s_test)

    x_test.reshape(x_test.shape + (1,))
    y_test, le = encode_labels(y_test, save_le=True)

    model = keras.models.load_model('trained_models/results/3_Jul_24/model.keras')

    # with open('label_encoder.pkl', 'rb') as pkl_le:
    #     le = pickle.load(pkl_le)

    predict_artist('test_songs/quincy_jones/0.wav', 'trained_models/results/3_Jul_24/model.keras', le)

    # plot_confusion_matrix(model = model, x_test=x_test, y_test=y_test, le=le)
    # x_test = x_test[:100]
    # y_test = y_test[:100]
    # s_test = s_test[:100]

    # print(x_test)
    # print(y_test)

    # print('Input Data Shape:', x_test.shape)

    # analyze_misclassificaitons(x_test=x_test, y_pred=model(x_test), y_test=y_test, s_test=s_test, le=le)