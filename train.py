import os
import numpy as np
import pandas as pd

import constants
from model import build_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

FEATURES_COUNT = 64
LABELS_COUNT = 7
BATCH_SIZE = 64
EPOCHS = 100


def load():
    X = np.load(constants.TRAIN_X)
    Y = np.load(constants.TRAIN_Y)
    return X, Y


def plot_random_images(X, count=10):
    import matplotlib.pyplot as plt
    for image in range(10):
        plt.figure(image)
        plt.imshow(X[image].reshape((constants.WIDTH, constants.HEIGHT)), interpolation='none', cmap='gray')
    plt.show()


def train(X, Y):
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=17)

    model = build_model(FEATURES_COUNT, LABELS_COUNT, constants.WIDTH, constants.HEIGHT)
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    model.fit(np.array(X_train), np.array(Y_train),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(Y_valid)),
          shuffle=True)

    model.save(constants.MODEL_PATH)
    print('Model is saved to {}'.format(constants.MODEL_PATH))


if __name__ == '__main__':
    X, Y = load()
    train(X, Y)
