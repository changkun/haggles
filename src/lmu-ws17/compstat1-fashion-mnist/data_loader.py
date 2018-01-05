# from keras.datasets import mnist
import os
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


TRAIN = os.path.join(os.path.dirname(__file__),
                     '../data/fashion-mnist/train.csv')
TEST = os.path.join(os.path.dirname(__file__),
                    '../data/fashion-mnist/test.csv')
random_state = 2018


def load_data():
    data = pd.read_csv(TRAIN, delimiter=',', dtype='|U',
                       quotechar='"', encoding="utf-8").values
    x = data[:, 1:]
    y = data[:, 0]
    x = x.astype(dtype=float)
    y = y.astype(dtype=float)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.05, random_state=random_state)
    return x_train, x_val, y_train, y_val


def load_test():
    data = pd.read_csv(TEST, delimiter=',', dtype='|U',
                       quotechar='"', encoding="utf-8").values
    x_id = data[:, 0]
    x = data[:, 1:]
    return x_id.astype(dtype=int), x.astype(dtype=float)


def preprocessing():
    x_train, x_val, y_train, y_val = load_data()
    x_id, x_test = load_test()

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255
    x_val = x_val / 255
    x_test = x_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    return x_train, x_val, y_train, y_val, x_id, x_test
