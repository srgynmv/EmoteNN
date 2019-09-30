import os
import numpy as np
import pandas as pd

from model import build_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


FEATURES_COUNT = 64
LABELS_COUNT = 7
BATCH_SIZE = 64
EPOCHS = 100
WIDTH, HEIGHT = 48, 48

DATA_X = 'data/data_x'
DATA_Y = 'data/data_y'
TEST_X = 'data/test_x'
TEST_Y = 'data/test_y'
MODEL_WEIGHTS = 'data/model_weights.h5'


def load():
    X = np.load(f'{DATA_X}.npy')
    Y = np.load(f'{DATA_Y}.npy')
    return X, Y


def plot_random_images(X, count=10):
    import matplotlib.pyplot as plt
    for image in range(10):
        plt.figure(image)
        plt.imshow(X[image].reshape((48, 48)), interpolation='none', cmap='gray')
    plt.show()


def train(X, Y):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=17)

    np.save(TEST_X, X_test)
    np.save(TEST_Y, Y_test)

    model = build_model(FEATURES_COUNT, LABELS_COUNT, WIDTH, HEIGHT)
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    model.fit(np.array(X_train), np.array(Y_train),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(Y_valid)),
          shuffle=True)

    model.save_weights(f'{MODEL_WEIGHTS}')
    print('Model is saved to {}.npy'.format(os.path.abspath(MODEL_WEIGHTS)))


if __name__ == '__main__':
    X, Y = load()
    train(X, Y)
