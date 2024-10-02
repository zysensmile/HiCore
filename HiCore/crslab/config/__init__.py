import os
from os.path import dirname, realpath

from .config import Config

ROOT_PATH = dirname(dirname(dirname(realpath(__file__))))
SAVE_PATH = os.path.join(ROOT_PATH, 'save')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
MODEL_PATH = os.path.join(DATA_PATH, 'model')
PRETRAIN_PATH = os.path.join(MODEL_PATH, 'pretrain')
EMBEDDING_PATH = os.path.join(DATA_PATH, 'embedding')
