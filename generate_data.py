import os
import zipfile
import numpy as np
import pandas as pd

DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
UNPACKED_DIR = os.path.join(DATASETS_DIR, 'unpacked')
FER_ARCHIVE_PATH = os.path.join(DATASETS_DIR, 'fer2013.zip')

FER_CSV_PATH = os.path.join(UNPACKED_DIR, 'fer2013', 'fer2013.csv')
FER_WIDTH = 48
FER_HEIGHT = 48

PREPROCESSED_DATA_DIR = 'data' 
DATA_X = 'data_x'
DATA_Y = 'data_y'


def unpack(archive_path):
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    unpack_path = os.path.join(UNPACKED_DIR, archive_name)

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

    if not os.path.isdir(PREPROCESSED_DATA_DIR):
        os.mkdir(PREPROCESSED_DATA_DIR)

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
    return X


if __name__ == "__main__":
    unpack(FER_ARCHIVE_PATH)
    X, Y = generate_data()
    X = preprocess_input(X)

    np.save(os.path.join(PREPROCESSED_DATA_DIR, DATA_X), X)
    np.save(os.path.join(PREPROCESSED_DATA_DIR, DATA_Y), Y)
