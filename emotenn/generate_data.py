import os
import zipfile
import numpy as np
import pandas as pd
from . import constants
from sklearn.model_selection import train_test_split


FER_ARCHIVE_PATH = os.path.join(constants.DATASETS_DIR, 'fer2013.zip')
FER_CSV_PATH = os.path.join(constants.UNPACKED_DIR, 'fer2013', 'fer2013.csv')
FER_WIDTH = 48
FER_HEIGHT = 48


def unpack(archive_path):
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    unpack_path = os.path.join(constants.UNPACKED_DIR, archive_name)

    if os.path.exists(unpack_path):
        return

    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extractall(unpack_path)


def generate_data():
    data = pd.read_csv(FER_CSV_PATH)
    pixels = data['pixels'].tolist()

    X = []
    for image in pixels:
        image = [int(pixel) for pixel in image.split(' ')]
        image = np.asarray(image).reshape(FER_WIDTH, FER_HEIGHT)
        X.append(image.astype('float32'))

    X = np.asarray(X)
    X = np.expand_dims(X, -1)

    Y = pd.get_dummies(data['emotion']).to_numpy()

    print(f'Loaded {len(X)} images')
    print(f'Image shape: {X[0].shape}')

    return X, Y


def preprocess_input(X, expand_range=True):
    """
    Preprocess images by scaling them between -1 to 1.
    """
    X = X.astype('float32')
    X = X / 255.0
    if expand_range:
        X -= 0.5
        X *= 2.0

    # No need to normalize a single image
    if np.size(X, 0) > 1:
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)

    return X


def save(X, Y, file_name):
    os.makedirs(constants.DATASETS_DIR, exist_ok=True)

    x_path = os.path.join(constants.DATASETS_DIR, file_name + '_x.npy')
    np.save(x_path, X)

    y_path = os.path.join(constants.DATASETS_DIR, file_name + '_y.npy')
    np.save(y_path, Y)

    print(f'Generated {len(X)} items')
    print(f'Data saved in {x_path} and {y_path}')


def main(args):
    unpack(FER_ARCHIVE_PATH)
    X, Y = generate_data()
    X = preprocess_input(X)
    save(X, Y, 'data')
