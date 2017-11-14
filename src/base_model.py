from keras.utils import np_utils
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


def base():
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=10, activation='sigmoid'))
    model.add(Dense(units=10, activation='sigmoid'))
    model.add(Dense(units=233, activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    return model


def base_r():
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=689, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units=689, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units=689, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def base_cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))
    return model


def base_cnn2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))
    model.summary()
    return model


def base_cnn3(input_shape=(28, 28, 1), classes=10):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same",
                     input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(
        3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
