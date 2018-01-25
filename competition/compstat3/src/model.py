from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from keras.regularizers import l2

def cifar10(input_shape=(32, 32, 3), classes=10):

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2*32, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*32, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4*32, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(4*32, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))

    return model