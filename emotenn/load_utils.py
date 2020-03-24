import os
import numpy as np
import requests
from . import constants


def create_dirs_for_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def download(src_url, dst_path):
    r = requests.get(src_url)
    create_dirs_for_path(dst_path)
    with open(dst_path, 'wb') as dst_file:
        dst_file.write(r.content)


# Thanks https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    create_dirs_for_path(destination)
    save_response_content(response, destination)


def load_dataset(dataset):
    x_path = os.path.join(constants.DATASETS_DIR, dataset.x.name)
    if not os.path.exists(x_path):
        download_file_from_google_drive(dataset.x.id, x_path)
    y_path = os.path.join(constants.DATASETS_DIR, dataset.y.name)
    if not os.path.exists(y_path):
        download_file_from_google_drive(dataset.y.id, y_path)
    X = np.load(x_path)
    Y = np.load(y_path)
    return X, Y
