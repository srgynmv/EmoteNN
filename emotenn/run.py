import os
import cv2
import sys
import argparse
import requests
from . import constants

import numpy as np
from tensorflow.keras.models import load_model
from .generate_data import preprocess_input


HAARCASCADE_PATH = os.path.join(constants.PREPROCESSED_DATA_DIR, 'haarcascade_frontalface_default.xml')
HAARCASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'


def download(src_url, dst_path):
    r = requests.get(src_url)
    directory = os.path.dirname(dst_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(dst_path, 'wb') as dst_file:
        dst_file.write(r.content)


class EmoteClassifier:
    def __init__(self, model_path=None):
        # Load keras model
        model_source = model_path or os.path.join(constants.TRAINED_MODELS_DIR, 'model.h5')
        self.model = load_model(model_source)

        # Download cascades
        if not os.path.exists(HAARCASCADE_PATH):
            download(HAARCASCADE_URL, HAARCASCADE_PATH)
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

    def classify(self, face):
        # Preprocess
        face = cv2.resize(face, constants.SIZE)
        face = np.reshape(face, (1, constants.WIDTH, constants.HEIGHT, 1))
        face = preprocess_input(face)

        # Infer
        classes = self.model.predict(face)
        return np.ravel(classes)

    def process_source(self, source, on_frame_callback):
        # Use video file path or video camera number as a source
        capture_source = source or 0
        capture = cv2.VideoCapture(capture_source)

        while capture.isOpened():
            ok, img = capture.read()
            if not ok:
                break

            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)

            # Find faces using the cascade
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_img, 1.1, 5)

            emotion_classes = []
            for x, y, w, h in faces:
                face = gray_img[y:y+h, x:x+w]
                emotion_classes.append(self.classify(face))

            processed_faces = zip(faces, emotion_classes)
            stopped = on_frame_callback(img, processed_faces, timestamp)

            if stopped:
                break

        capture.release()


class WindowOutput:
    WINDOW_SIZE = 960, 540
    WINDOW_NAME = 'Emotion recognition'
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1
    DRAW_COLOR = (255, 255, 0)
    BORDER_SIZE = 2
    FONT_THICKNESS = 2
    LABEL_PADDING = 10

    def __init__(self, draw_probabilites=False):
        self.draw_probabilities = draw_probabilites

    def open(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    def close(self):
        cv2.destroyAllWindows()

    def on_frame(self, img, faces, timestamp):
        # Draw face box and labels
        for (x, y, w, h), emotion_classes in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), self.DRAW_COLOR, self.BORDER_SIZE)
            if self.draw_probabilities:
                labels = ['{}: {:.2f}'.format(label, emotion) for label, emotion in zip(constants.CLASS_NAMES, emotion_classes)]
                self.draw_labels(img, x, y, w, h, labels)
            else:
                class_idx = np.argmax(emotion_classes)
                label = constants.CLASS_NAMES[class_idx]
                self.draw_label(img, x, y, w, h, label)

        # Display a final image
        cv2.imshow(self.WINDOW_NAME, img)
        cv2.waitKey(1)

        # Stop the video processing when the window is closed
        return cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1

    @classmethod
    def draw_labels(cls, img, x, y, w, h, labels):
        label_y = y
        for idx, label in enumerate(labels):
            (_, label_height), baseline = cv2.getTextSize(label,
                                                          cls.FONT_FACE,
                                                          cls.FONT_SCALE,
                                                          cls.FONT_THICKNESS)
            label_y += label_height
            if idx > 0:
                label_y += baseline

            cv2.putText(img,
                        label,
                        (x+w+cls.LABEL_PADDING, label_y),
                        cls.FONT_FACE,
                        cls.FONT_SCALE,
                        cls.DRAW_COLOR,
                        cls.FONT_THICKNESS)

    @classmethod
    def draw_label(cls, img, x, y, w, h, label):
        (label_width, _), baseline = cv2.getTextSize(label,
                                                     cls.FONT_FACE,
                                                     cls.FONT_SCALE,
                                                     cls.FONT_THICKNESS)
        label_x = int(x + (w - label_width) / 2)
        cv2.putText(img,
                    label,
                    (label_x, y - baseline - cls.LABEL_PADDING),
                    cls.FONT_FACE,
                    cls.FONT_SCALE,
                    cls.DRAW_COLOR,
                    cls.FONT_THICKNESS)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def fill_arguments(parser):
    parser.add_argument('source', nargs='?', default=None, help='path to a source for the emotion recognition')
    parser.add_argument('--model', default=None, help='path to a trained model')
    parser.add_argument('--probs', action='store_true', help='display probabilities of all classes instead of one class')


def main(args):
    classifier = EmoteClassifier(model_path=args.model)
    with WindowOutput(draw_probabilites=args.probs) as output:
        classifier.process_source(args.source, output.on_frame)
