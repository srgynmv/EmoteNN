import sys
import constants
import numpy as np
from keras.models import load_model


def load_test_data():
    X = np.load(constants.TEST_X)
    Y = np.load(constants.TEST_Y)
    return X, Y


def main():
    if len(sys.argv) < 1:
        print('Please pass the path to a model!')
        return

    model = load_model(sys.argv[1])
    model.summary()
    X_test, Y_test = load_test_data()
    score = model.evaluate(X_test, Y_test)
    print("{}: {:.2f}%".format(model.metrics_names[1], score[1] * 100))


if __name__ == "__main__":
    main()