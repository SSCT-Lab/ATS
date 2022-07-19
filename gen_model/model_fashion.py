#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
val_loss: 0.2939
val_acc: 0.8988
'''

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import np_utils


# # LeNet-5
# def model_fashion():
#     ### modify
#     # path='./fashion-mnist/data/fashion'
#     # X_train, y_train = mnist_reader.load_mnist(path, kind='train')
#     # X_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
#     # X_train = X_train.astype('float32').reshape(-1,28,28,1)
#     # X_test = X_test.astype('float32').reshape(-1,28,28,1)
#     ### modify
#     (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()  ### modify
#     X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
#     X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
#     X_train /= 255
#     X_test /= 255
#     ### modify
#     print('Train:{},Test:{}'.format(len(X_train), len(X_test)))
#     nb_classes = 10
#     y_train = np_utils.to_categorical(y_train, nb_classes)
#     y_test = np_utils.to_categorical(y_test, nb_classes)
#     print('data success')
#     input_tensor = Input((28, 28, 1))
#     # 28*28
#     temp = Conv2D(filters=6, kernel_size=(5, 5), padding='valid', use_bias=False)(input_tensor)
#     temp = Activation('relu')(temp)
#     # 24*24
#     temp = MaxPooling2D(pool_size=(2, 2))(temp)
#     # 12*12
#     temp = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', use_bias=False)(temp)
#     temp = Activation('relu')(temp)
#     # 8*8
#     temp = MaxPooling2D(pool_size=(2, 2))(temp)
#     # 4*4
#     # 1*1
#     temp = Flatten()(temp)
#     temp = Dense(120, activation='relu')(temp)
#     temp = Dense(84, activation='relu')(temp)
#     output = Dense(nb_classes, activation='softmax')(temp)
#     model = Model(input=input_tensor, outputs=output)
#     model.summary()
#     sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     checkpoint = ModelCheckpoint(filepath='./model/model_fashion.hdf5', monitor='val_acc', mode='auto',
#                                  save_best_only='True')
#     model.fit(X_train, y_train, batch_size=64, nb_epoch=15, validation_data=(X_test, y_test), callbacks=[checkpoint])
#     model = load_model('./model/model_fashion.hdf5')
#     score = model.evaluate(X_test, y_test, verbose=0)
#     print(score)


# LeNet-1
# [0.2967624181032181, 0.8971999883651733]
def model_fashion_LeNet_1(X_train, Y_train, X_test, Y_test, filepath='./model/model_fashion_LeNet1.hdf5'):
    nb_classes = 10
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    print('data success')

    input_tensor = Input((28, 28, 1))
    temp = Conv2D(4, (5, 5), padding='same')(input_tensor)
    temp = Activation('relu')(temp)
    temp = MaxPooling2D(pool_size=(2, 2))(temp)

    temp = Conv2D(12, (5, 5), padding='same')(temp)
    temp = Activation('relu')(temp)
    temp = MaxPooling2D(pool_size=(2, 2))(temp)

    temp = Flatten()(temp)
    output = Dense(10, activation='softmax')(temp)
    model = Model(input=input_tensor, outputs=output)
    model.summary()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, Y_train, batch_size=64, nb_epoch=15, validation_data=(X_test, Y_test), callbacks=[checkpoint],
              verbose=True)
    model = load_model(filepath)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)


if __name__ == '__main__':
    model_fashion_LeNet_1()
