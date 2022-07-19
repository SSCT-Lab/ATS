from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.models import Model, load_model
from keras.utils import np_utils


# vgg-16
def model_cifar_vgg16(X_train, Y_train, X_test, Y_test, filepath='./model/model_cifar_vgg16.hdf5', nb_epoch=30, ):
    ### modify
    print('Train:{},Test:{}'.format(len(X_train), len(X_test)))
    nb_classes = 10
    y_train = np_utils.to_categorical(Y_train, nb_classes)
    y_test = np_utils.to_categorical(Y_test, nb_classes)
    print('data success')
    # base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(28, 28, 1))
    model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    # for layer in model_vgg.layers:  # 冻结权重
    #     layer.trainable = False
    model = Flatten()(model_vgg.output)
    model = Dense(1024, activation='relu', name='fc1')(model)
    # model = Dropout(0.5)(model)
    model = Dense(512, activation='relu', name='fc2')(model)
    # model = Dropout(0.5)(model)
    model = Dense(10, activation='softmax', name='prediction')(model)
    model = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pretrain')
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, validation_data=(X_test, y_test),
              callbacks=[checkpoint])
    model = load_model(filepath)
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)
