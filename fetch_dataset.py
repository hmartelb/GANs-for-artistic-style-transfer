import os
import zipfile
from six.moves import urllib

DATASET_DOWNLOAD_URL = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/21755/1475600/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1604920625&Signature=f%2F4LGkhXM6r4oL4MqZ4YvQUx7s4%2BtQ5m3gEbvhPUGZkbQcnxJRGn4KEg4%2FzkbPBzrP4NEtP1b5zcqmlfbnoMnb0Al0YnfiQpzrQgRzGTdVweSaker1TytOHedhvJof8JrF6Lt1Pj2cMNxT7W2wbfJ00FACtR3Bd0GXVZ3AM1ZfhxQP1%2Bddz2CLwACRhF9QbBs%2B90SkbyCoU933gJ7jsvqoojbnlSNV8g4vcKl%2BNBSWd9mdiRbkzBNAZMrUjC5GeRKhKw7BiZ4Pobveu69bJ2c5yMfRpap7GHLX1QgocBqKDlDGxokFT%2Bis6uBNwue5QwLIyNuZ%2FWFstuP0b0veJ2rQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dgan-getting-started.zip"

DATASET_DIR = "./data"
DATASET_ZIP = "dataset.zip"

def fetch_dataset():
    if not os.path.isdir(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    zip_path = os.path.join(DATASET_DIR, DATASET_ZIP)
    urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
        zip_ref.close()
    os.remove(zip_path)

if __name__ == "__main__":
    fetch_dataset()