import os

# paths section
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_DIR = os.path.join(PROJECT_ROOT_DIR, 'datasets')
UNPACKED_DIR = os.path.join(DATASETS_DIR, 'unpacked')
PREPROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'trained_models')

TRAIN_X = os.path.join(PREPROCESSED_DATA_DIR, 'train_x.npy')
TRAIN_Y = os.path.join(PREPROCESSED_DATA_DIR, 'train_y.npy')
TEST_X = os.path.join(PREPROCESSED_DATA_DIR, 'test_x.npy')
TEST_Y = os.path.join(PREPROCESSED_DATA_DIR, 'test_y.npy')
CHECKPOINTS_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'checkpoints')

# data section
WIDTH = 48
HEIGHT = 48
SIZE = (WIDTH, HEIGHT)
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
