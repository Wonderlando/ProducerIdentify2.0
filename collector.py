import wave
from pydub import AudioSegment
import pyaudio
import pyautogui
from PIL import ImageChops
from threading import Thread
import time
import ffmpeg
import os

from dataclasses import dataclass, asdict

# Default parameters for streaming
@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000 # 44100 before changing
    frames_per_buffer: int = 1024
    input: bool = True
    output: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

class Recorder:
    """Recorder using blocking I/O facility from pyaudio to record sound
    
    Attributes:
        - stream_params: StreamParams object with values for pyaudio Stream object"""
    
    def __init__(self, region: tuple, stream_params: StreamParams) -> None:
        self.stream_params = stream_params
        self._region = region
        self._pyaudio = None
        self._stream = None
        self._wav_file = None
        self._capturing = False
        self._screen_capture = None

    def record(self, save_path: str) -> None:
        """Record sound from mic for a given amount of seconds.
        
        :param save_path: Where to store recording
        """
        self._create_recording_resources(save_path)
        self._initialize_threads()

    def _create_recording_resources(self, save_path: str) -> None:
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(**self.stream_params.to_dict())
        self._create_wav_file(save_path)

    def _create_wav_file(self, save_path: str):
        self._wav_file = wave.open(save_path, "wb") # python built in write wave file
        self._wav_file.setnchannels(self.stream_params.channels)
        self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.stream_params.format))
        self._wav_file.setframerate(self.stream_params.rate)

    def _write_wav_file_reading_from_stream(self) -> None:
        self._capturing = True
        print('recording...')
        while self._capturing == True:
            audio_data = self._stream.read(self.stream_params.frames_per_buffer)
            self._wav_file.writeframes(audio_data)

    def finish_recording(self) -> None:
        print('Finishing up')
        self._stream.close()
        self._pyaudio.terminate()

    # Main function to continuously monitor a specific area for changes
    def _monitor_screen_changes(self):
        time.sleep(10) # ensure that you don't skip while the play animation goes
        print('monitoring screen changes...')
        self._screen_capture = None
        while True:
            # must convert to type "L" (b/w) because alpha value does not allow proper PIL comparison
            current_screenshot = pyautogui.screenshot(region = self._region).convert('L')
            if self._screen_capture is not None:
                if ImageChops.difference(self._screen_capture, current_screenshot).getbbox():
                    print('stopping recording')
                    self._capturing = False
                    break
            self._screen_capture = current_screenshot

    def _initialize_threads(self):

        threads = (
            Thread(target=self._write_wav_file_reading_from_stream),
            Thread(target = self._monitor_screen_changes)
        )

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

if __name__ == "__main__":

    record_songs = True
    fix_songs = False

    producer_name = 'daft_punk'
    music_folder = 'music'
    songs_in_album = 300

    song_length_tolerance = 45

    producer_directory = f'./{music_folder}/{producer_name}'

    if record_songs:
        stream_params = StreamParams()
        region_to_monitor = (100, 800, 400, 100)  # Region of title of song
        recorder = Recorder(region=region_to_monitor, stream_params=stream_params)


        if os.path.isdir(producer_directory):
            print(f'Producer directory found for {producer_name}')
        else:
            print(f'Producer directory not found. Creating new directory for {producer_name}')
            os.mkdir(producer_directory)

        for n in range(songs_in_album):
            save_path = f'{producer_directory}/{n}.wav'
            recorder.record(save_path = save_path)

        recorder.finish_recording()

    if fix_songs:
        # check if audio is less than 30 sec
        for n in range(len(os.listdir(producer_directory))):
            with wave.open(f'{producer_directory}/{n}.wav', 'r') as wav_file:
                # Get the number of frames in the file
                frames = wav_file.getnframes()
                # Get the frame rate (number of frames per second)
                rate = wav_file.getframerate()
            # Calculate the duration in seconds
            duration = frames / float(rate)
            if duration < song_length_tolerance:
                print(f'Song {n} is less than {song_length_tolerance} seconds! Combining with song {n-1} and deleting song {n}')
                # load appropriate songs
                audio1 = AudioSegment.from_wav(f'{producer_directory}/{n-1}.wav')
                audio2 = AudioSegment.from_wav(f'{producer_directory}/{n}.wav')

                # Combine the audio files
                combined_audio = audio1 + audio2
                combined_audio.export(f'{producer_directory}/{n-1}.wav', format="wav")

                # delete old file
                os.remove(f'{producer_directory}/{n}.wav')