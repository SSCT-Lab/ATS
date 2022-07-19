import os

import numpy as np

from gen_data.DauUtils import get_dau
from utils import model_conf
from utils.utils import shuffle_data

cov_name_list = ["NAC", "TKNC", "NBC", "SNAC"]
dau_name_arr = ['SF', 'ZM', 'BR', 'RT', 'CT', 'BL', 'SR']
mop_name_arr = ["IR", "ORI", "RG", "RP", "CR"]
rank_name_list = ["CES", "Gini", "DeepDiv", "Random", "MAX_P", "LSA"]



def mk_exp_dir(exp_name, data_name, model_name, base_path):
    pair_name = model_conf.get_pair_name(data_name, model_name)
    dir_name = exp_name + "_" + pair_name
    base_path = base_path + "/" + dir_name
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    return base_path


def _get_IR_dau(data_name):
    if data_name == model_conf.mnist:
        dau = get_dau(model_conf.fashion)
    elif data_name == model_conf.fashion:
        dau = get_dau(model_conf.mnist)
    elif data_name == model_conf.cifar10:
        dau = get_dau(model_conf.svhn)
    elif data_name == model_conf.svhn:
        dau = get_dau(model_conf.cifar10)
    else:
        raise ValueError()
    return dau


def get_mutation_data(op, mop, x_dau, y_dau, data_name, seed=0):
    ratio = 0.2
    if op == "LB":
        x_arr = []
        y_arr = []
        for x_dop, y_dop in zip(x_dau, y_dau):
            x_lb, y_lb = mop.label_bias(x_dop, y_dop, seed=seed)
            x_arr.append(x_lb)
            y_arr.append(y_lb)
        x_select = np.concatenate(x_arr, axis=0)
        y_select = np.concatenate(y_arr, axis=0)
    else:
        x_select = np.concatenate(x_dau, axis=0)
        y_select = np.concatenate(y_dau, axis=0)
        if op == "RG":
            img_shape = x_select[0].shape
            x_add, y_add = mop.synthetic_data(len(x_select), img_shape, ratio=ratio, seed=seed)
        elif op == "RP":
            x_add, y_add = mop.repeat_data(x_select, y_select, ratio=ratio, seed=seed)
        elif op == "IR":
            dau = _get_IR_dau(data_name)
            (x_extra, _), (_, _) = dau.load_data(use_norm=True)
            x_add, y_add = mop.irrelevant_data(len(x_select), x_extra, ratio=ratio, seed=seed)
            del _
        elif op == "CR":
            x_add, y_add = mop.corruption_data(x_select, y_select, ratio=ratio, seed=seed)
        else:
            raise ValueError()
        print(op, "add", len(x_add))
        x_select = np.concatenate([x_select, x_add], axis=0)
        y_select = np.concatenate([y_select, y_add], axis=0)
    x_select, y_select = shuffle_data(x_select, y_select, 0)
    assert len(x_select) == len(y_select)
    return x_select, y_select


def get_dau_data(x_test, y_test, dau, dau_name_arr, ratio=0.5, shuffle=False):
    x_test_arr = []
    y_test_arr = []

    x_val_dict = {}
    y_val_dict = {}
    num = int(len(x_test) * ratio)
    x_test_arr.append(x_test[:num])
    y_test_arr.append(y_test[:num])
    x_val_dict["ORI"] = x_test[num:]
    y_val_dict["ORI"] = y_test[num:]

    for dau_op_name in dau_name_arr:
        # print(dau_op_name)
        x, y = dau.load_dau_data(dau_op_name, use_norm=True, use_cache=False)
        if shuffle:
            x, y = shuffle_data(x, y, 0)
        num = int(len(x) * ratio)
        x_test_arr.append(x[:num])
        y_test_arr.append(y[:num])

        x_val_dict[dau_op_name] = x[num:]
        y_val_dict[dau_op_name] = y[num:]


    return x_test_arr, y_test_arr, x_val_dict, y_val_dict
