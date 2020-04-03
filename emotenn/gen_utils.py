import os
import zipfile
import numpy as np
from . import constants


def unpack(archive_path):
    archive_name = os.path.splitext(os.path.basename(archive_path))[0]
    unpack_path = os.path.join(constants.UNPACKED_DIR, archive_name)

    if os.path.exists(unpack_path):
        return

    with zipfile.ZipFile(archive_path, 'r') as archive:
        archive.extractall(unpack_path)


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

    file_path = os.path.join(constants.DATASETS_DIR, file_name + '.bin')
    with open(file_path, 'wb') as np_file:
        np.save(np_file, X)
        np.save(np_file, Y)

    print(f'Generated {len(X)} items')
    print(f'Data saved in {file_path}')
