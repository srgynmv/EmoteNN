import os
import sys
import numpy as np
import pandas as pd
import pickle
import importlib.util

import constants
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import callbacks
from sklearn.model_selection import train_test_split


FEATURES_COUNT = 64
LABELS_COUNT = 7
BATCH_SIZE = 64
EPOCHS = 100


def load(model_path):
    spec = importlib.util.spec_from_file_location("model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model(FEATURES_COUNT, LABELS_COUNT, constants.WIDTH, constants.HEIGHT)

    X = np.load(constants.TRAIN_X)
    Y = np.load(constants.TRAIN_Y)
    return model, X, Y


def plot_random_images(X, count=10):
    import matplotlib.pyplot as plt
    for image in range(10):
        plt.figure(image)
        plt.imshow(X[image].reshape((constants.WIDTH, constants.HEIGHT)), interpolation='none', cmap='gray')
    plt.show()


def get_callbacks():
    if not os.path.isdir(constants.CHECKPOINTS_DIR):
        os.mkdir(constants.CHECKPOINTS_DIR)
    cb_list = []
    cb_list.append(callbacks.ModelCheckpoint(os.path.join(constants.CHECKPOINTS_DIR, 'model-{epoch:02d}-{val_loss:.2f}.h5'), period=5))
    cb_list.append(callbacks.EarlyStopping(patience=5))
    return cb_list


def train(model, X, Y):
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=17)

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    history = model.fit(np.array(X_train), np.array(Y_train),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(np.array(X_valid), np.array(Y_valid)),
          shuffle=True,
          callbacks=get_callbacks())

    return history


def save(result_filename, model, history):
    if not os.path.isdir(constants.TRAINED_MODELS_DIR):
        os.mkdir(constants.TRAINED_MODELS_DIR)

    model_file_path = os.path.join(constants.TRAINED_MODELS_DIR, result_filename) + '.h5'
    model.save(model_file_path)
    print('Model is saved to {}'.format(model_file_path))

    history_path = os.path.join(constants.TRAINED_MODELS_DIR, result_filename) + '_history.pkl'
    with open(history_path, 'wb') as history_file:
        pickle.dump(history.history, history_file)
    print('Train history is saved to {}'.format(history_path))


def main():
    if len(sys.argv) < 1:
        print('Please pass the path to a model!')
        return

    model_path = sys.argv[1]
    model, X, Y = load(model_path)
    history = train(model, X, Y)
    result_filename = os.path.basename(model_path).split('.')[0]
    save(result_filename, model, history)


if __name__ == '__main__':
    main()
