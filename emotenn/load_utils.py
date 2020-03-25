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


def download_file_from_google_drive(gdrive_file, exist_ok=False):
    # Thanks https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
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

    destination = gdrive_file.path
    if exist_ok and os.path.exists(destination):
        return

    id = gdrive_file.id
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
    download_file_from_google_drive(dataset.x, exist_ok=True)
    download_file_from_google_drive(dataset.y, exist_ok=True)
    X = np.load(dataset.x.path)
    Y = np.load(dataset.y.path)
    return X, Y
