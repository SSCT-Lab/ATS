import os
from keras.engine.saving import load_model
from tqdm import tqdm

import exp_utils
from exp_utils import mk_exp_dir, get_dau_data
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from utils import model_conf
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')


def exec(model_name, data_name, rank_type):
    if rank_type == "rank":
        rank_name_list = exp_utils.rank_name_list
        base_dir = "result_raw/Div{}".format("")
        suffix = ""
    elif rank_type == "cov":
        rank_name_list = exp_utils.cov_name_list
        base_dir = "result_raw/Cov{}".format("")
        suffix = "_EXT"
    else:
        raise ValueError()
    base_path = mk_exp_dir(exp_name, data_name, model_name, base_dir)
    exp_repeat(model_name, data_name, base_path, rank_name_list, suffix=suffix)


def exp_repeat(model_name, data_name, base_path, rank_name_list, suffix=""):
    dau_name_arr = exp_utils.dau_name_arr
    mop_name_arr = exp_utils.mop_name_arr
    print(dau_name_arr)
    dau = get_dau(data_name)
    mop = Mop()
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    del x_train
    ps_path = "{}/ps_data/".format(base_path)
    diverse_error_dir = "{}/diverse_error/".format(base_path)
    fault_dir = "{}/fault/".format(base_path)
    time_path = "{}/time/".format(base_path)
    os.makedirs(time_path, exist_ok=True)
    os.makedirs(ps_path, exist_ok=True)
    os.makedirs(diverse_error_dir, exist_ok=True)
    os.makedirs(fault_dir, exist_ok=True)
    select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1]
    split_ratio = 0.5
    sample_num = 1
    model_path = model_conf.get_model_path(data_name, model_name)
    print(model_path)
    ori_model = load_model(model_path)
    ############################
    # exp
    ############################

    x_dau_test_arr, y_dau_test_arr, x_val_dict, y_val_dict = get_dau_data(x_test, y_test, dau, dau_name_arr,
                                                                          ratio=split_ratio,
                                                                          shuffle=True)

    select_size_arr = []
    len_data = len(np.concatenate(y_dau_test_arr, axis=0))
    for select_size_ratio in select_size_ratio_arr:
        select_size = int(len_data * select_size_ratio)
        select_size_arr.append(select_size)
    max_select_size = select_size_arr[-1]
    print("max_select_size", max_select_size)
    for mop_name in mop_name_arr:
        for i in range(sample_num):
            if mop_name == "ORI":
                x_select = np.concatenate(x_dau_test_arr, axis=0)
                y_select = np.concatenate(y_dau_test_arr, axis=0)
            else:
                x_select, y_select = exp_utils.get_mutation_data(mop_name, mop, x_dau_test_arr, y_dau_test_arr,
                                                                 data_name, seed=i)
            save_path = os.path.join(ps_path, "{}_" + "{}_{}.npy".format(mop_name, i))
            diverse_error_path = os.path.join(diverse_error_dir, "{}_" + "{}_{}.npy".format(mop_name, i))
            fault_path = os.path.join(fault_dir, "{}_" + "{}_{}.npy".format(mop_name, i))
            print("len(x_select)", len(x_select))

            idx_data = {}
            for name in rank_name_list:
                fp_extra = save_path.format(name + suffix)
                if os.path.exists(fp_extra):
                    ix = np.load(fp_extra)
                else:
                    raise ValueError()
                assert len(ix) >= max_select_size
                idx_data[name] = ix
            print("mop_name :{}".format(mop_name), "keys:", idx_data.keys(), "select_size_arr:", select_size_arr)
            for k, idx in tqdm(idx_data.items()):
                diverse_error_idx_path = diverse_error_path.format(k)
                fault_idx_path = fault_path.format(k)
                x_s, y_s = x_select[idx][:max_select_size], y_select[idx][:max_select_size]
                save_diverse_error_idx(diverse_error_idx_path, ori_model, x_s, y_s)
                save_fault_idx_path(fault_idx_path, ori_model, x_s, y_s, mop_name)

            total_diverse_error_idx_path = os.path.join(diverse_error_dir, "_" + "{}_{}.npy".format(mop_name, i))
            total_fault_idx_path = os.path.join(fault_dir, "_" + "{}_{}.npy".format(mop_name, i))
            save_diverse_error_idx(total_diverse_error_idx_path, ori_model, x_select, y_select)
            save_fault_idx_path(total_fault_idx_path, ori_model, x_select, y_select, mop_name)
            del x_select


def save_diverse_error_idx(error_idx_path, ori_model, x_s, y_s):
    y_prob = ori_model.predict(x_s)
    del x_s
    y_psedu = np.argmax(y_prob, axis=1)
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
    np.save(error_idx_path, fault_idx_arr)


def save_fault_idx_path(error_idx_path, ori_model, x_s, y_s, mop_name):
    y_prob = ori_model.predict(x_s)
    y_psedu = np.argmax(y_prob, axis=1)
    fault_idx_arr = []
    for ix, (x_s_temp, y_s_temp, y_psedu_temp) in enumerate(zip(x_s, y_s, y_psedu)):
        if y_s_temp == -1:
            continue
        elif y_s_temp == y_psedu_temp:
            continue
        else:
            if mop_name == "RP":
                add_flag = True
                for select_ix in fault_idx_arr:
                    if (x_s_temp == x_s[select_ix]).all():
                        add_flag = False
                        break
                if add_flag:
                    fault_idx_arr.append(ix)
            else:
                fault_idx_arr.append(ix)
    np.save(error_idx_path, fault_idx_arr)


if __name__ == '__main__':
    exp_name = "exp_mutation"  # exp_repeat
    only_add = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    for data_name, v_arr in tqdm(model_conf.model_data.items()):
        for model_name in v_arr:
            print(model_name, data_name)
            for rank_type in ["rank", "cov"]:
                exec(model_name, data_name, rank_type=rank_type)
