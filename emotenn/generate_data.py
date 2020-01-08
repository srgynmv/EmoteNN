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


def save(X, Y):
    if not os.path.isdir(constants.PREPROCESSED_DATA_DIR):
        os.mkdir(constants.PREPROCESSED_DATA_DIR)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    
    np.save(constants.TRAIN_X, X_train)
    np.save(constants.TRAIN_Y, Y_train)
    np.save(constants.TEST_X, X_test)
    np.save(constants.TEST_Y, Y_test)

    print(f'{len(X_train)} train images, {len(X_test)} test images')


def main():
    unpack(FER_ARCHIVE_PATH)
    X, Y = generate_data()
    X = preprocess_input(X)
    save(X, Y)


if __name__ == "__main__":
    main()