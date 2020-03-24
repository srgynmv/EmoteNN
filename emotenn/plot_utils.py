import os
import pickle
import numpy as np
import seaborn as sns
from . import constants
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


def get_class_name(y):
    names = np.array(constants.CLASS_NAMES)
    return names[y == 1][0]


def plot_model_history(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    # Plot training & validation accurnacy values
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_random_images(X, Y, count=10):
    fig, axes = plt.subplots(1, count)
    for ax in axes:
        image_idx = np.random.randint(0, len(X))
        ax.imshow(X[image_idx].reshape((constants.WIDTH, constants.HEIGHT)), interpolation='none', cmap='gray')
        ax.set_title(get_class_name(Y[image_idx]))
        ax.axis('off')
    plt.show()


def draw_data_metrics(data_x, data_y):
    print('Data dimensions: {}'.format(data_x.shape)

    data_y = [np.where(vec == 1)[0][0] for vec in data_y] # convert from 1-hot vec to class index
    classes, counts = np.unique(data_y, return_counts=True)

    # Draw bars
    plt.figure(dpi=100)
    plot = plt.bar(classes, height=counts) 
    # Draw bar values
    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2.,
             1.002*height,'%d' % int(height), ha='center', va='bottom')
    # Draw class names
    plt.xticks(classes, constants.CLASS_NAMES)
    # Remove the chart frame
    plt.box(False)
    plt.show()


def plot_confusion_matrix(model, test_x, test_y):
    Y_prediction = model.predict(test_x)

    # Convert classification results to one hot vectors 
    Y_pred_classes = np.argmax(Y_prediction, axis = 1) 
    Y_true = np.argmax(test_y, axis=1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', xticklabels=constants.CLASS_NAMES, yticklabels=constants.CLASS_NAMES)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_model_graph(model):
    plot_model(model, show_shapes=True, show_layer_names=True)
