import os
from collections import namedtuple

# paths section
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_DIR = os.path.join(PROJECT_ROOT_DIR, 'datasets')
UNPACKED_DIR = os.path.join(DATASETS_DIR, 'unpacked')
RESULTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'trained_models')
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

# data section
WIDTH = 48
HEIGHT = 48
SIZE = (WIDTH, HEIGHT)
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# urls
GDriveFile = namedtuple('GDriveFile', ['id', 'name'])
Dataset = namedtuple('Dataset', ['x', 'y'])

FER2013 = Dataset(
    x=GDriveFile(id='1ixw7odMh7jTOHgTQBRISd_WUdsvOU0P2', name='fer2013_x.npy'),
    y=GDriveFile(id='1VjpSF_fzjWQ78yiiNwbzVYuhLrM3S9rQ', name='fer2013_y.npy')
)