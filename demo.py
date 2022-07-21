import os

import keras
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from termcolor import colored

from ATS.ATS import ATS
from gen_data.MnistDau import MnistDau
from utils import model_conf
import numpy as np

from utils.utils import num_to_str, shuffle_data, shuffle_data3


def color_print(s, c):
    print(colored(s, c))


class DemoMnistDau(MnistDau):
    def get_dau_params(self):
        params = {
            "SF": [(0, 0.15), (0, 0.15)],
        }
        return params


def train_model(model, filepath, X_train, Y_train, X_test, Y_test, epochs=10, verbose=1):
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_test, Y_test),
              callbacks=[checkpoint],
              verbose=verbose)
    model = load_model(filepath)
    return model


def get_psedu_label(m, x):
    pred_test_prob = m.predict(x)
    y_test_psedu = np.argmax(pred_test_prob, axis=1)
    return y_test_psedu


def diverse_errors_num(y_s, y_psedu):
    fault_pair_arr = []
    fault_idx_arr = []
    for ix, (y_s_temp, y_psedu_temp) in enumerate(zip(y_s, y_psedu)):
        if y_s_temp == -1:
            continue
        elif y_s_temp == y_psedu_temp:
            continue
        else:
            key = (y_s_temp, y_psedu_temp)
            if key not in fault_pair_arr:
                fault_pair_arr.append(key)
                fault_idx_arr.append(ix)
    return len(fault_idx_arr)


def get_tests(x_dau, y_dau):
    x_sel, y_sel = x_dau[:test_size // 2], y_dau[:test_size // 2]
    x_val, y_val = x_dau[test_size // 2:], y_dau[test_size // 2:]
    return x_sel, y_sel, x_val, y_val


def fault_detection(y, y_psedu):
    fault_num = np.sum(y != y_psedu)
    print("fault num : {}".format(fault_num))

    diverse_fault_num = diverse_errors_num(y, y_psedu)
    print("diverse_fault_num  : {}/{}".format(diverse_fault_num, 90))
    return fault_num, diverse_fault_num


def retrain(model_path, x, y, base_path):
    M = load_model(model_path)
    filepath = os.path.join(base_path, "temp.h5")
    trained_model = train_model(M, filepath, x,
                                keras.utils.np_utils.to_categorical(y, 10), x_val,
                                keras.utils.np_utils.to_categorical(y_val, 10))
    acc_val1 = trained_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10))[1]
    print("retrain model path: {}".format(filepath))
    print("train acc improve {} -> {}".format(acc_val0, acc_val1))
    return acc_val1


# a demo for ATS
if __name__ == '__main__':
    # initial ATS
    base_path = "demo"
    os.makedirs(base_path, exist_ok=True)

    ats = ATS()

    # mnist data
    color_print("load LeNet-5 model and MNIST data sets", "blue")
    dau = DemoMnistDau()
    (x_train, _), (x_test, y_test) = dau.load_data(use_norm=True)
    test_size = dau.test_size
    nb_classes = model_conf.fig_nb_classes

    # LeNet5 model
    model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet5)
    ori_model = load_model(model_path)

    acc = ori_model.evaluate(x_test, keras.utils.np_utils.to_categorical(y_test, 10), verbose=0)[1]
    print("ori test accuracy {}".format(acc))

    # data augmentation
    color_print("data augmentation", "blue")
    # dau.run("test")
    x_dau, y_dau = dau.load_dau_data("SF", use_cache=False)
    x_dau, y_dau = shuffle_data(x_dau, y_dau)

    # selection
    color_print("adaptive test selection on the augmented data", "blue")
    x_sel, y_sel, x_val, y_val = get_tests(x_dau, y_dau)
    acc_val0 = ori_model.evaluate(x_val, keras.utils.np_utils.to_categorical(y_val, 10), verbose=0)[1]

    y_sel_psedu = get_psedu_label(ori_model, x_sel)
    div_rank, _, _ = ats.get_priority_sequence(x_sel, y_sel_psedu, nb_classes, ori_model, th=0.001)

    xs, ys, ys_psedu = x_sel[div_rank], y_sel[div_rank], y_sel_psedu[div_rank]
    print("priority sequence : {} ...".format(div_rank[:10]))

    # 1000
    color_print("Select the first 1000 augmented data", "blue")
    num = 1000
    # ATS
    xs_num, ys_num, ys_psedu_num = xs[:num], ys[:num], ys_psedu[:num]
    # Random
    xr, yr, yr_psedu = shuffle_data3(x_sel, y_sel, y_sel_psedu)
    xr_num, yr_num, yr_psedu_num = xr[:num], yr[:num], yr_psedu[:num]

    # fault detection
    color_print("fault detection on selected data", "blue")
    print("ATS")
    fault_num, diverse_fault_num = fault_detection(ys_num, ys_psedu_num)
    print("Random")
    fault_num2, diverse_fault_num2 = fault_detection(yr_num, yr_psedu_num)
    color_print("fault detection difference between ATS and Random :{}".format(fault_num - fault_num2), "green")
    color_print(
        "diverse fault detection difference between ATS and Random :{}".format(diverse_fault_num - diverse_fault_num2),
        "green")

    # retrain
    # ATS
    color_print("retrain model on selected data", "blue")
    print("ATS")
    acc_val1 = retrain(model_path, xs_num, ys_num, base_path)
    print("Random")
    acc_val2 = retrain(model_path, xr_num, yr_num, base_path)

    color_print("accuracy difference between ATS and Random :{}".format(acc_val1 - acc_val2), "green")
