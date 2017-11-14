import numpy as np
import os
import csv
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras import backend as K

import base_model
from data_loader import load_data, load_test, preprocessing


batch_size = 128
epochs = 50
input_shape = (28, 28, 1)
model_weights = os.path.join(os.path.dirname(
    __file__),  '../model/fashion-mnist/cnn.h5')
model_path = os.path.join(os.path.dirname(
    __file__),  '../model/fashion-mnist/cnn.h5')
submission_path = os.path.join(os.path.dirname(
    __file__), '../data/fashion-mnist/submission.csv')


def mobilenet():
    # TODO: Fix here
    input_image = Input(shape=(128, 128))
    input_image_ = Lambda(lambda x: K.repeat_elements(
        K.expand_dims(x, 3), 3, 3))(input_image)
    print(input_image_)
    base_model = MobileNet(input_tensor=input_image_,
                           include_top=False, weights='imagenet', pooling='avg')
    output = Dropout(0.5)(base_model.output)
    predict = Dense(10, activation='softmax')(output)
    model = Model(inputs=input_image, outputs=predict)
    model.summary()
    return model


# 1. preprocessing
x_train, x_val, y_train, y_val, x_id, x_test = preprocessing()

# 2. model
model = None
if os.path.isfile() and os.path.isfile(model_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)
    print('Loaded model from disk')
else:
    model = base_model.base_cnn3()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    gen = ImageDataGenerator(rotation_range=8,
                             width_shift_range=0.08,
                             shear_range=0.3,
                             height_shift_range=0.08,
                             zoom_range=0.08)
    val_gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_gen.flow(x_val, y_val, batch_size=batch_size)

    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0.0001,
                      patience=5, verbose=2, mode='auto'),
        ModelCheckpoint('fashion-mnist.h5', monitor='val_loss',
                        save_best_only=True, verbose=0)
    ]

    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    model.fit_generator(train_generator,
                        steps_per_epoch=x_train.shape[0] // 64,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=x_val.shape[0] // 64,
                        callbacks=callbacks)

    train_results = model.evaluate(x_train, y_train)
    val_results = model.evaluate(x_val, y_val)

    model_json = model.to_json()
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(model_weights)
    print('Saved model to disk')

    print('Train       Loss: ', train_results[0])
    print('Train       Acc: ',  train_results[1])
    print('Validation  Loss:',  val_results[0])
    print('Validation  Acc: ',  val_results[1])

# 3. generate submission
y_ = model.predict(x_test)
y_ = np.argmax(y_, axis=1)

x_id = x_id.reshape(x_id.shape[0], 1)
y_ = y_.reshape(y_.shape[0], 1)
results = np.concatenate((x_id, y_), axis=1)
np.savetxt(submission_path, results, '%d', delimiter=',')
