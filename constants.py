import os

# paths section
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(THIS_DIR, 'datasets')
UNPACKED_DIR = os.path.join(DATASETS_DIR, 'unpacked')
PREPROCESSED_DATA_DIR = os.path.join(THIS_DIR, 'data') 

TRAIN_X = os.path.join(PREPROCESSED_DATA_DIR, 'train_x.npy')
TRAIN_Y = os.path.join(PREPROCESSED_DATA_DIR, 'train_y.npy')
TEST_X = os.path.join(PREPROCESSED_DATA_DIR, 'test_x.npy')
TEST_Y = os.path.join(PREPROCESSED_DATA_DIR, 'test_y.npy')
MODEL_PATH = os.path.join(PREPROCESSED_DATA_DIR, 'model.h5')

# data section
WIDTH = 48
HEIGHT = 48