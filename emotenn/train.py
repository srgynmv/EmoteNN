import os
import pickle
from . import constants
from tensorflow.keras import callbacks


def get_callbacks(model_filename):
    if not os.path.isdir(constants.CHECKPOINTS_DIR):
        os.mkdir(constants.CHECKPOINTS_DIR)

    cb_list = []
    cb_list.append(callbacks.ModelCheckpoint(os.path.join(constants.CHECKPOINTS_DIR, model_name + '-{epoch:02d}-{val_loss:.2f}.h5'), period=5))
    cb_list.append(callbacks.EarlyStopping(monitor='val_accuracy', patience=25))
    return cb_list


def save_results(model, history, model_filename):
    if not os.path.isdir(constants.TRAINED_MODELS_DIR):
        os.mkdir(constants.TRAINED_MODELS_DIR)

    model_file_path = os.path.join(constants.TRAINED_MODELS_DIR, model_filename) + '.h5'
    model.save(model_file_path)
    print('Model is saved to {}'.format(model_file_path))

    history_path = os.path.join(constants.TRAINED_MODELS_DIR, model_filename) + '_history.pkl'
    with open(history_path, 'wb') as history_file:
        pickle.dump(history.history, history_file)
    print('Train history is saved to {}'.format(history_path))
