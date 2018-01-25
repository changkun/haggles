from os.path import join, dirname
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

data_train = f'{dirname(__file__)}/../data/train.csv'
data_pred = f'{dirname(__file__)}/../data/test.csv'


def load_data(dataset_size=0.1, test_size=0.1, random_state=42):
    data = pd.read_csv(
        data_train, delimiter=',', dtype='|U', encoding='utf-8'
    ).values
    x = data[:, 1:].astype(dtype=float)
    y = data[:, 0].astype(dtype=float) - 1
    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=test_size, random_state=random_state
    )

    pred = pd.read_csv(
        data_pred,
        delimiter=',', dtype='|U',
        quotechar='"', encoding='utf-8'
    ).values
    x_id = pred[:, -1].astype(dtype=int)
    x_pred = pred[:, :-1].astype(dtype=float)

    # TODO: on loading phase
    size_train = int(x_train.shape[0] * dataset_size)
    size_val   = int(x_val.shape[0] * dataset_size)
    size_pred  = int(x_pred.shape[0] * dataset_size)

    # preprocessing
    x_train = x_train[:size_train]
    x_val   = x_val[:size_val]
    x_pred  = x_pred[:size_pred]

    y_train = np_utils.to_categorical(y_train)[:size_train]
    y_val   = np_utils.to_categorical(y_val)[:size_val]

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_val   = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_pred  = x_pred.reshape(x_pred.shape[0], 28, 28, 1)
    x_id    = x_id[:size_pred]

    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)

    return (x_train, y_train), (x_val, y_val), (x_pred, x_id)
