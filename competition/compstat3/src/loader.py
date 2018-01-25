from os.path import join, dirname
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

data_train = f'{dirname(__file__)}/../data/train.csv'
data_pred = f'{dirname(__file__)}/../data/test.csv'


def load_data(dataset_size=0.1, test_size=0.1, random_state=42):
    img_width  = 32
    img_height = 32
    channels = 3

    data = pd.read_csv(
        data_train,
        delimiter=',', dtype='|U', encoding='utf-8'
    ).values

    x = data[:, :-1].astype('float32')
    y = data[:, -1].astype('float32')
    pred = pd.read_csv(
        data_pred,
        delimiter=',', dtype='|U', encoding='utf-8'
    ).values
    x_id = pred[:, -1].astype(dtype=int)
    x_pred = pred[:, :-1].astype(dtype=float)

    x = x.reshape(
        -1, channels, img_width*img_height
    ).transpose(0, 2, 1).reshape(
        -1, img_width, img_height, channels
    )
    x_pred  = x_pred.reshape(
        -1, channels, img_width*img_height
    ).transpose(0, 2, 1).reshape(
        -1, img_width, img_height, channels
    )

    mean = np.mean(x,axis=(0, 1, 2, 3))
    std = np.std(x,axis=(0, 1, 2, 3))
    x = (x - mean) / (std + 1e-7)
    x_pred = (x_pred - mean) /(std + 1e-7)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=test_size, random_state=random_state
    )

    # TODO: on loading phase
    size_train = int(x_train.shape[0] * dataset_size)
    size_val   = int(x_val.shape[0] * dataset_size)
    size_pred  = int(x_pred.shape[0] * dataset_size)

    y_train = np_utils.to_categorical(y_train)[:size_train]
    y_val   = np_utils.to_categorical(y_val)[:size_val]

    x_train = x_train[:size_train]
    x_val   = x_val[:size_val]
    x_pred  = x_pred[:size_pred]
    x_id    = x_id[:size_pred]

    return (x_train, y_train), (x_val, y_val), (x_pred, x_id)