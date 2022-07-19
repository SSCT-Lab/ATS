import os
import time

import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras import backend as K


def retrain_detail(temp_model_path, x_s, y_s, X_train, Y_train, x_val_dict, y_val_dict, model_path, nb_classes,
                   verbose=1, only_add=False):
    # 1 . 合并训练集
    if only_add:
        Ya_train = y_s
        Xa_train = x_s
    else:
        Ya_train = np.concatenate([Y_train, y_s])
        Xa_train = np.concatenate([X_train, x_s])
    # 2. hot
    Ya_train_vec = keras.utils.np_utils.to_categorical(Ya_train, nb_classes)

    # 2. 加载模型
    ori_model = load_model(model_path)

    # 验证集
    x_val_arr = []
    y_val_arr = []
    # val_base_dict = {}
    for op, x_val in x_val_dict.items():
        y_val = y_val_dict[op]
        y_val_vec = keras.utils.np_utils.to_categorical(y_val, nb_classes)
        x_val_arr.append(x_val)
        y_val_arr.append(y_val)
        # acc_si_val_ori = ori_model.evaluate(x_val, y_val_vec, verbose=0)[1]
        # val_base_dict[op] = acc_si_val_ori

    X_val = np.concatenate(x_val_arr, axis=0)
    Y_val = np.concatenate(y_val_arr, axis=0)
    Y_val_vec = keras.utils.np_utils.to_categorical(Y_val, nb_classes)

    # 在 验证集上的精度  泛化鲁邦性
    acc_base_val = ori_model.evaluate(X_val, Y_val_vec, verbose=0)[1]
    sss = time.time()

    trained_model = retrain_model(temp_model_path, ori_model, Xa_train, Ya_train_vec, X_val, Y_val_vec, 0,
                                  verbose=verbose)
    eee = time.time()
    acc_si_val = trained_model.evaluate(X_val, Y_val_vec, verbose=0)[1]

    # acc_si_val_var = cal_acc_var(X_val, Y_val, nb_classes, trained_model)

    acc_imp_val = acc_si_val - acc_base_val
    val_si_dict = {}
    val_si_dict["all"] = acc_imp_val
    # for op, x_val in x_val_dict.items():
    #     y_val = y_val_dict[op]
    #     y_val_vec = keras.steps.np_utils.to_categorical(y_val, nb_classes)
    #     acc_si_val_op = trained_model.evaluate(x_val, y_val_vec, verbose=0)[1]
    #     val_si_dict[op] = acc_si_val_op - val_base_dict[op]

    print("val acc", acc_base_val, acc_si_val, "diff:", format(acc_imp_val, ".3f"))
    K.clear_session()  # 每次重训练后都清缓存
    del trained_model
    del ori_model
    del Xa_train
    del x_val_arr
    del X_val
    return val_si_dict, eee - sss


def retrain_model(temp_path, ori_model, x_si, y_si_vector, Xa_test, Ya_test_vec, idx=0, verbose=1):
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    new_model_name = str(idx) + "_.hdf5"
    filepath = "{}/{}".format(temp_path, new_model_name)
    trained_model = train_model(ori_model, filepath, x_si, y_si_vector, Xa_test, Ya_test_vec, verbose=verbose)
    return trained_model


def train_model(model, filepath, X_train, Y_train, X_test, Y_test, epochs=15, verbose=1):
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_test, Y_test),
              callbacks=[checkpoint],
              verbose=verbose)
    model = load_model(filepath)
    return model


def get_retrain_csv_data(name, imp_dict, time, ):
    csv_data = {
        "name": name,
        "time": time,
    }
    for op, x_val in imp_dict.items():
        csv_data[op] = imp_dict[op]
    return csv_data
