import os
import pickle
import tensorflow as tf
from . import constants
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split


def get_callbacks(model_name):
    os.makedirs(constants.CHECKPOINTS_DIR, exist_ok=True)

    cb_list = []
    cb_list.append(callbacks.ModelCheckpoint(os.path.join(constants.CHECKPOINTS_DIR, model_name + '-{epoch:02d}-{val_loss:.2f}.h5'), period=5))
    cb_list.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=25))
    return cb_list


def split_dataset(data_x, data_y, valid_size, test_size, align_by=None):
    train_size = 1.0 - valid_size - test_size
    if align_by is not None:
        train_size = int(train_size * len(data_x))
        train_size -= train_size % align_by
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=train_size)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=test_size / (test_size + valid_size)) 
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def save_results(model, history, model_name):
    os.makedirs(constants.TRAINED_MODELS_DIR, exist_ok=True)

    model_file_path = os.path.join(constants.TRAINED_MODELS_DIR, model_name) + '.h5'
    model.save(model_file_path)
    print('Model is saved to {}'.format(model_file_path))

    history_path = os.path.join(constants.TRAINED_MODELS_DIR, model_name) + '_history.pkl'
    with open(history_path, 'wb') as history_file:
        pickle.dump(history.history, history_file)
    print('Train history is saved to {}'.format(history_path))


def get_distribution_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    except ValueError:
        strategy = tf.distribute.MirroredStrategy()
    return strategy
