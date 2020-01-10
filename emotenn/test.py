import sys
import numpy as np
from . import constants
from tensorflow.keras.models import load_model


def load_test_data():
    X = np.load(constants.TEST_X)
    Y = np.load(constants.TEST_Y)
    return X, Y


def fill_arguments(parser):
    parser.add_argument('model', help='path to a model')


def main(args):
    model = load_model(args.model)
    model.summary()
    X_test, Y_test = load_test_data()
    score = model.evaluate(X_test, Y_test)
    print("{}: {:.2f}%".format(model.metrics_names[1], score[1] * 100))
