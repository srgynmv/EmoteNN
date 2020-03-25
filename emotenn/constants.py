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

FER_ARCHIVE = GDriveFile('1Syp3_xi0rV_DlWYQP4KVYmfH9f4djtzT', os.path.join(DATASETS_DIR, 'fer2013.zip'))
FER2013 = Dataset(
    x=GDriveFile('1ixw7odMh7jTOHgTQBRISd_WUdsvOU0P2', os.path.join(DATASETS_DIR, 'fer2013_x.npy')),
    y=GDriveFile('1VjpSF_fzjWQ78yiiNwbzVYuhLrM3S9rQ', os.path.join(DATASETS_DIR, 'fer2013_y.npy'))
)

LU_CNN = GDriveFile('1RHaMKknF6ues_QPqxpQsXCrFRNScBbOA', os.path.join(TRAINED_MODELS_DIR, 'lu_cnn.h5'))
LU_CNN_HISTORY = GDriveFile('1M4up-LLmMMRaq7XDlrD8CpermiOZrWg8', os.path.join(TRAINED_MODELS_DIR, 'lu_cnn_history.pkl'))
MODEL = GDriveFile('1tygLOWx0vj9pjvUqQkETXrPPciVHygOG', os.path.join(TRAINED_MODELS_DIR, 'model.h5'))
MODEL_HISTORY = GDriveFile('1CFobUGWj2zdSKR-mLMA8SQnpV20VBkDP', os.path.join(TRAINED_MODELS_DIR, 'model_history.pkl'))
