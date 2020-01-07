import os
import cv2
import sys
import argparse
import requests
import constants

import numpy as np
from keras.models import load_model
from generate_data import preprocess_input


HAARCASCADE_PATH = os.path.join(constants.PREPROCESSED_DATA_DIR, 'haarcascade_frontalface_default.xml')
HAARCASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'

WINDOW_SIZE = 960, 540
WINDOW_NAME = 'Emotion recognition'
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
DRAW_COLOR = (255, 255, 0)
BORDER_SIZE = 2
FONT_THICKNESS = 2
LABEL_PADDING = 10


def download(src_url, dst_path):
    r = requests.get(src_url)
    directory = os.path.dirname(dst_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(dst_path, 'wb') as dst_file:
        dst_file.write(r.content)


def classify(model, face):
    # Preprocess
    face = cv2.resize(face, constants.SIZE)
    face = np.reshape(face, (1, constants.WIDTH, constants.HEIGHT, 1))
    face = preprocess_input(face)

    # Infer
    classes = model.predict(face)
    return np.ravel(classes)


def draw_labels(img, x, y, w, h, labels):
    label_y = y
    for idx, label in enumerate(labels):
        (_, label_height), baseline = cv2.getTextSize(label,
                                                      FONT_FACE,
                                                      FONT_SCALE,
                                                      FONT_THICKNESS)
        label_y += label_height
        if idx > 0:
            label_y += baseline

        cv2.putText(img,
                    label,
                    (x+w+LABEL_PADDING, label_y),
                    FONT_FACE,
                    FONT_SCALE,
                    DRAW_COLOR,
                    FONT_THICKNESS)


def draw_label(img, x, y, w, h, label):
    (label_width, _), baseline = cv2.getTextSize(label,
                                                 FONT_FACE,
                                                 FONT_SCALE,
                                                 FONT_THICKNESS)
    label_x = int(x + (w - label_width) / 2)
    cv2.putText(img,
                label,
                (label_x, y - baseline - LABEL_PADDING),
                FONT_FACE,
                FONT_SCALE,
                DRAW_COLOR,
                FONT_THICKNESS)


def mainloop(source_path=None, model_path=None, draw_probabilities=False):
    # Download cascades
    if not os.path.exists(HAARCASCADE_PATH):
        download(HAARCASCADE_URL, HAARCASCADE_PATH)
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    
    # Use video file path or video camera number as a source
    capture_source = source_path or 0
    capture = cv2.VideoCapture(capture_source)

    # Load keras model
    model_source = model_path or os.path.join(constants.TRAINED_MODELS_DIR, 'model.h5')
    model = load_model(model_source)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while capture.isOpened():
        ok, img = capture.read()
        if not ok:
            break

        # Find faces using the cascade
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)
    
        for x, y, w, h in faces:
            face = gray_img[y:y+h, x:x+w]
            emotion_classes = classify(model, face)
            # Draw face box and labels
            cv2.rectangle(img, (x, y), (x+w, y+h), DRAW_COLOR, BORDER_SIZE)
            if draw_probabilities:
                labels = ['{}: {:.2f}'.format(label, emotion) for label, emotion in zip(constants.CLASS_NAMES, emotion_classes)]
                draw_labels(img, x, y, w, h, labels)
            else:
                class_idx = np.argmax(emotion_classes)
                label = constants.CLASS_NAMES[class_idx]
                draw_label(img, x, y, w, h, label)

        # Display a final image
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(1)

        # Break when the window is closed
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    capture.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='?', default=None, help='path to a source for the emotion recognition')
    parser.add_argument('--model', default=None, help='path to a trained model')
    parser.add_argument('--probs', action='store_true', help='display probabilities of all classes instead of one class')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mainloop(source_path=args.source,
             model_path=args.model,
             draw_probabilities=args.probs)
