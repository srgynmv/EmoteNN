import sys
import time
import pyaudio
import librosa
import librosa.display
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from threading import Lock


# Audio stream settings
CHUNK = 2048
RATE = 44100
FORMAT = pyaudio.paFloat32


# MFCC settings
MFCC_TIME = 2.5
SAMPLES = int(RATE * MFCC_TIME)
SILENCE_THRESHOLD = 1e-2


# Labels
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def normalize_mfcc(mfcc):
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    mfcc = mfcc.transpose()
    mfcc -= mean
    mfcc /= std
    return mfcc.transpose()


def preprocess_lstm(mfcc):
    # Transform dimensions to the form of (count, timesteps, features)
    norm_mfcc = np.reshape(norm_mfcc, (1, *mfcc.shape))
    norm_mfcc = np.asarray(norm_mfcc, dtype=np.float32)
    return np.swapaxes(norm_mfcc, 1, 2)


def preprocess_cnn(mfcc):
    return np.reshape(mfcc, (1, *mfcc.shape, 1))


def get_emotion(model, mfcc):
    classes = model.predict(mfcc)
    class_idx = np.argmax(np.ravel(classes))
    return CLASS_NAMES[class_idx]


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time demonstration of the audio emotion recognition')
    parser.add_argument('--model', default='', help='path to the trained model')
    parser.add_argument('--type', choices=['rnn', 'cnn'], help='type of the model, required to make the correct preprocessing')
    parser.add_argument('--show-mfcc', action='store_true', help='display mfcc on the plot')
    parser.add_argument('--use-absolute-mfcc', action='store_true', help='use absolute values of mfcc instead of normalized')
    return parser.parse_args()


def main(args):
    update_buffer_lock = Lock()

    # Configure the pyplot window
    mpl.rcParams['toolbar'] = 'None'
    mpl.rcParams['figure.autolayout'] = True
    plt.ion()
    fig = plt.figure(figsize=(6,3))

    waveplot_ax = plt.subplot(211) if args.show_mfcc else plt.subplot(111)
    mfcc_ax = plt.subplot(212) if args.show_mfcc else None

    window_visible = True
    fig.canvas.set_window_title('Audio Emotion Recognition')

    def handle_close(event):
        nonlocal window_visible
        window_visible = False

    fig.canvas.mpl_connect('close_event', handle_close)

    # Init the audio buffer 
    buffer = np.array([], dtype=np.float32)
    def on_stream_callback(stream_data, frames_count, time_info, status_flags):
        nonlocal buffer
        data = np.fromstring(stream_data, dtype=np.float32)
        with update_buffer_lock:
            buffer = np.concatenate((buffer, data))
            buffer = buffer[-SAMPLES:]
        return None, pyaudio.paContinue if window_visible else pyaudio.paComplete

    # Open the audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=on_stream_callback)

    # Load the model
    model = tf.keras.models.load_model(args.model)
    preprocess_fn = preprocess_cnn if args.type == 'cnn' else preprocess_lstm
    mfcc_number = model.input.shape[1] if args.type == 'cnn' else model.input.shape[2]

    try: 
        while window_visible:
            with update_buffer_lock:
                last_buffer_slice = buffer.copy()

            if last_buffer_slice.size < SAMPLES:
                mfcc = None
                status = 'Gathering data...'
            else:
                mfcc = librosa.feature.mfcc(y=last_buffer_slice, sr=RATE, n_mfcc=mfcc_number)
                if not args.use_absolute_mfcc:
                    mfcc = normalize_mfcc(mfcc)

                if max(last_buffer_slice) >= SILENCE_THRESHOLD:
                    status = get_emotion(model, preprocess_fn(mfcc))
                else:
                    status = 'Waiting for a speech...'

            # Draw waveplot
            waveplot_ax.cla()
            waveplot_ax.set_title(status)
            waveplot_ax.set_ylim(-0.5, 0.5)
            librosa.display.waveplot(last_buffer_slice, sr=RATE, ax=waveplot_ax)

            if mfcc_ax and mfcc is not None:
                mfcc_ax.cla()
                mfcc_ax.imshow(mfcc, aspect='auto', interpolation='none')

            plt.pause(0.0001)
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main(parse_args())