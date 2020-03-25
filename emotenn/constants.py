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

CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# urls
GDriveFile = namedtuple('GDriveFile', ['id', 'path'])
Dataset = namedtuple('Dataset', ['x', 'y'])

FER2013 = Dataset(
    x=GDriveFile('1ixw7odMh7jTOHgTQBRISd_WUdsvOU0P2', 
                 os.path.join(DATASETS_DIR, 'fer2013_x.npy')),
    y=GDriveFile('1VjpSF_fzjWQ78yiiNwbzVYuhLrM3S9rQ', 
                 os.path.join(DATASETS_DIR, 'fer2013_y.npy'))
)
