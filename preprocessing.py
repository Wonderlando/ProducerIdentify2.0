'''This file should serve to pre-process wave files that come in and export it in a clean fashion

We want to export the raw audio file, track isolation, song characteristics. 

'''

import numpy as np
import librosa
import logging
import os
import dill
from pydub import AudioSegment

def create_dataset(producer_folder='music', save_folder='dataset',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (producer_folder/producer/*.wav)
    and saves it to a specified folder."""

    logging.basicConfig(filename = 'error.log', level = logging.ERROR)

    if not os.path.isdir(save_folder):
        print(f'Dataset folder not found. Creating folder \'{save_folder}\'')
        os.makedirs(save_folder, exist_ok=True)
    # get list of all producers
    producers = [path for path in os.listdir(producer_folder) if
               os.path.isdir(os.path.join(producer_folder, path))]

    # iterate through all songs and create mel spectrogram
    for producer in producers:
        print(producer)
        producer_path = os.path.join(producer_folder, producer)
        producer_songs = [song for song in os.listdir(producer_path)]

        for song in producer_songs:
            song_path = os.path.join(producer_path, song)

            try:
                # Create mel spectrogram and convert it to the log scale
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
                    continue

            # check length of audiofile for potential errors
            if librosa.get_duration(y=y, sr=sr) < 45:
                print(f"Song {str(song_path)} is less than 45 seconds. Check audio file to ensure this is not a data issue.")

            try:
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length)
                log_S = librosa.power_to_db(S, ref=1.0)
                data = (producer, log_S, song)

                # Save each song
                save_name = producer + '_%%-%%_' + '_%%-%%_' + song
                with open(os.path.join(save_folder, save_name), 'wb') as fp:
                    dill.dump(data, fp)
            except Exception as e:
                print(f"Error processing {song_path}: {e}")

if __name__ == '__main__':
    create_dataset(producer_folder='music', save_folder='dataset',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512)