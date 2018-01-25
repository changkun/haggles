from loader import load_data
from model import cifar10

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import adam, rmsprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

class Hyperparams:
    def __init__(self):
        # TODO: configularion file
        self.pretrain = False
        self.server = False
        self.architecture = '../model/architecture.json'
        self.inter_model = [
            '../model/intermediate_ep75.h5',
            '../model/intermediate_ep100.h5',
            '../model/intermediate_ep125.h5'
        ]
        self.submission  = '../data/submission.csv'
        self.batch_size = 128
        self.epochs = 25
        self.input_shape = (28, 28, 1)
        self.classe = 101


def main():
    params = Hyperparams()

    if params.server:
        (x_train, y_train), (x_val, y_val), (x_pred, x_id) = load_data(dataset_size=1.0)
    else:
        (x_train, y_train), (x_val, y_val), (x_pred, x_id) = load_data(dataset_size=0.1)

    model = cifar10()
    model.summary()

    if params.pretrain:
        model.load_weights(params.inter_model)
    else:
        generator_train = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
        ).flow(x_train, y_train, batch_size=params.batch_size)

        # search phase 1
        model.compile(
            loss=categorical_crossentropy,
            optimizer=rmsprop(lr=0.001,decay=1e-6),
            metrics=['accuracy']
        )
        history = model.fit_generator(
            generator_train,
            steps_per_epoch=x_train.shape[0] // params.batch_size,
            epochs=3*params.epochs,
            validation_data=(x_val, y_val)
        )
        model.save_weights(params.inter_model[0])

        # search phase 2
        model.compile(
            loss=categorical_crossentropy,
            optimizer=rmsprop(lr=0.0005,decay=1e-6),
            metrics=['accuracy']
        )
        model.fit_generator(
            generator_train,
            steps_per_epoch=x_train.shape[0] // params.batch_size,
            epochs=params.epochs,
            validation_data=(x_val, y_val)
        )
        model.save_weights(params.inter_model[1])

        # search phase 3
        model.compile(loss=categorical_crossentropy,
                optimizer=rmsprop(lr=0.0003,decay=1e-6),
                metrics=['accuracy'])
        model.fit_generator(
            generator_train,
            steps_per_epoch=x_train.shape[0] // params.batch_size,
            epochs=params.epochs,
            validation_data=(x_val, y_val))
        model.save_weights(params.inter_model[2])

    # model performance
    train_results = model.evaluate(x_train, y_train)
    val_results = model.evaluate(x_val, y_val)
    print('Train       Loss: ', train_results[0])
    print('Train       Acc: ',  train_results[1])
    print('Validation  Loss:',  val_results[0])
    print('Validation  Acc: ',  val_results[1])
    
    # saving prediction
    y_pred = model.predict(x_pred)
    y_pred = np.argmax(y_pred, axis=1) + 1
    x_id = x_id.reshape(x_id.shape[0], 1)
    y_pred = y_pred.reshape(y_pred.shape[0], 1)
    results = np.concatenate((x_id, y_pred), axis=1)
    np.savetxt(params.submission, results, '%d', delimiter=',')


if __name__ == '__main__':
    main()