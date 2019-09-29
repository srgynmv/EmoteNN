import os
import numpy as np
import pandas as pd

FER_CSV_PATH = 'datasets/fer2013/fer2013.csv'
FER_WIDTH = 48
FER_HEIGHT = 48

PREPROCESSED_DATA_DIR = 'data' 
DATA_X = 'data_x'
DATA_Y = 'data_y'


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


# It is a standard way to pre-process images by scaling them between -1 to 1. 
# Images is scaled to [0,1] by dividing it by 255. 
# Further, subtraction by 0.5 and multiplication by 2 changes the range to [-1,1]. 
# [-1,1] has been found a better range for neural network models in computer vision problems.
def preprocess_input(X, expand_range=True):
    X = X.astype('float32')
    X = X / 255.0
    if expand_range:
        X -= 0.5
        X *= 2.0
    return X


if __name__ == "__main__":
    X, Y = generate_data()
    X = preprocess_input(X)

    np.save(f'{PREPROCESSED_DATA_DIR}/{DATA_X}', X)
    np.save(f'{PREPROCESSED_DATA_DIR}/{DATA_Y}', Y)