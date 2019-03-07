import logging
import sys
from os.path import join

import matplotlib.image as img
import numpy as np
import yaml
from keras.preprocessing.sequence import pad_sequences
from skimage.transform import resize


def get_config():
    return yaml.load(open("config/config.yaml"))


config = get_config()
img_root = config["img_root"]
min_side = config["min_side"]
maxlen = config["maxlen"]


def get_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


image_cache = {}


def load_image(img_name):
    if not img_name in image_cache:
        try:
            img_arr = img.imread(join(img_root, img_name))[:, :, 0:3]
            # load the image and resize it accordingly
            img_arr_rs = resize(img_arr, (min_side, min_side, 3), mode='constant')
        except:
            # some images are not food image like (400,400), we use 0 for them
            img_arr_rs = np.zeros(shape=(299, 299, 3))
        image_cache[img_name] = img_arr_rs
    return image_cache[img_name]


def add_text_field(df):
    df['text'] = df['name'].fillna('') + ' ' + df['menu_name'].fillna('') + ' ' + df['item_description'].fillna('')


# tokenizing text
def vectorize_text(text_arr, tokenizer):
    tokenized_text = tokenizer.texts_to_sequences(text_arr)
    return pad_sequences(tokenized_text, maxlen=maxlen)
