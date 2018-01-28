
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

def base():
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=10, activation='sigmoid'))
    model.add(Dense(units=10, activation='sigmoid'))
    model.add(Dense(units=233, activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))
    return model

def fashion(input_shape=(28, 28, 1), classes=10):
    model = Sequential()

    model.add(Conv2D(
        32, (3, 3), padding='same', input_shape=input_shape,
        kernel_initializer='lecun_normal',
        activation='selu'
    ))
    model.add(BatchNormalization())
    model.add(Conv2D(
        32, (3, 3), padding='same', input_shape=input_shape,
        kernel_initializer='lecun_normal',
        activation='selu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(
        64, (3, 3), padding='same',
        kernel_initializer='lecun_normal',
        activation='selu'
    ))
    model.add(BatchNormalization())
    model.add(Conv2D(
        128, (3, 3), padding='same',
        kernel_initializer='lecun_normal',
        activation='selu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(
        128, (3, 3), padding='same',
        kernel_initializer='lecun_normal',
        activation='selu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='selu', kernel_initializer='lecun_normal'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Dense(classes, activation='softmax'))
    
    return model
