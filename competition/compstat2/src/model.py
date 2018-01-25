from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization
from keras.models import Model

def caltech101():
    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(16, (3, 3),
               padding='same',
               kernel_regularizer='l2',
               kernel_initializer='lecun_normal')(inputs)
    x = Activation(activation='selu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3),
               padding='same',
               kernel_regularizer='l2',
               kernel_initializer='lecun_normal')(x)
    x = Activation(activation='selu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3),
               padding='same',
               kernel_regularizer='l2',
               kernel_initializer='lecun_normal')(x)
    x = Activation(activation='selu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3),
               padding='same',
               kernel_regularizer='l2',
               kernel_initializer='lecun_normal')(x)
    x = Activation(activation='selu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), kernel_initializer='lecun_normal')(x)
    x = Activation(activation='selu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation(activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(101)(x)
    y = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=y)
    return model