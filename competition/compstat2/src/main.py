from loader import load_data
from model import caltech101

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

class Hyperparams:
    def __init__(self):
        # TODO: configularion file
        self.pretrain = False
        self.fullset = True
        self.architecture = '../model/architecture.json'
        self.inter_model = '../model/intermidiate.h5'
        self.submission  = '../data/submission.csv'
        self.batch_size = 128
        self.epochs = 100
        self.input_shape = (28, 28, 1)
        self.classe = 101


def main():
    params = Hyperparams()

    if params.fullset:
        (x_train, y_train), (x_val, y_val), (x_pred, x_id) = load_data(dataset_size=1.0)
    else:
        (x_train, y_train), (x_val, y_val), (x_pred, x_id) = load_data(dataset_size=0.1)

    model = caltech101()
    model.summary()

    if params.pretrain:
        model.load_weights(params.inter_model)
        model.compile(
            loss=categorical_crossentropy,
            optimizer=adam(lr=1e-3),
            metrics=['accuracy']
        )
    else:
        generator_train = ImageDataGenerator(
            rotation_range=30,
            shear_range=0.3,
            width_shift_range=0.08,
            height_shift_range=0.08,
            zoom_range=0.08,
            horizontal_flip=True,
            vertical_flip=True,
        ).flow(x_train, y_train, batch_size=params.batch_size)
        
        callbacks = [
            # EarlyStopping(
            #     monitor='val_loss', min_delta=0.0001,
            #     patience=5, verbose=1, mode='auto'
            # ),
            ModelCheckpoint(
                params.inter_model, monitor='val_loss',
                save_best_only=True, verbose=1
            )
        ]

        model.compile(
            loss=categorical_crossentropy,
            optimizer=adam(lr=1e-3),
            metrics=['accuracy']
        )
        history = model.fit_generator(
            generator_train,
            x_train.shape[0] // params.batch_size,
            epochs=params.epochs,
            validation_data=(x_val, y_val),
            validation_steps=x_val.shape[0] // params.epochs,
            callbacks=callbacks
        )

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