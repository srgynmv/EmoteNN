from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2


def build_model(features_count, labels_count, input_width, input_height):
    model = Sequential()

    model.add(Conv2D(features_count, kernel_size=(5, 5), activation='relu', input_shape=(input_width, input_height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(4*features_count, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4*features_count, activation='relu'))
    model.add(Dense(2*features_count, activation='relu'))
    model.add(Dense(features_count, activation='relu'))
    model.add(Dense(labels_count, activation='softmax'))

    return model
