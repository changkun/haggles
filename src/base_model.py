import keras
from keras.utils import np_utils
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.applications.mobilenet import MobileNet
from keras import backend as K


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


def mobilenet():
    input_image = Input(shape=(28, 28))
    input_image_ = Lambda(lambda x: K.repeat_elements(
        K.expand_dims(x, 3), 3, 3))(input_image)
    base_model = MobileNet(input_tensor=input_image_,
                           include_top=False, weights='imagenet', pooling='avg')
    output = Dropout(0.5)(base_model.output)
    predict = Dense(10, activation='softmax')(output)
    model = Model(inputs=input_image, outputs=predict)
    model.summary()
    return model


def resnet():
    input_size = (28, 28, 1)
    num_filters = 64
    num_blocks = 3
    num_sub_blocks = 2

    # Creating model based on ResNet published archietecture
    inputs = Input(shape=input_size)
    x = Conv2D(num_filters, padding='same',
               kernel_initializer='he_normal',
               kernel_size=7, strides=2,
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Check by applying max pooling later (setting it false as size of image is small i.e. 28x28)
    # x = MaxPooling2D(pool_size=3, padding='same', strides=2)(x)
    # Creating Conv base stack

    # Instantiate convolutional base (stack of blocks).
    for i in range(num_blocks):
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            # Creating residual mapping using y
            y = Conv2D(num_filters,
                       kernel_size=3,
                       padding='same',
                       strides=strides,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(num_filters,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(y)
            y = BatchNormalization()(y)
            if is_first_layer_but_not_first_block:
                x = Conv2D(num_filters,
                           kernel_size=1,
                           padding='same',
                           strides=2,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
            # Adding back residual mapping
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)

        num_filters = 2 * num_filters

    # Add classifier on top.
    x = AveragePooling2D()(x)
    y = Flatten()(x)
    outputs = Dense(10, activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate and compile model.
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
