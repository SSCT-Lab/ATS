import os

from tqdm import tqdm

import exp_utils
from exp_utils import mk_exp_dir
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from utils.train_utils import retrain_detail, get_retrain_csv_data
from utils import model_conf
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')
from utils.utils import add_df
from keras import backend as K


def exec(model_name, data_name, base_dir):
    print(model_name, data_name)
    base_path = mk_exp_dir(exp_name, data_name, model_name, base_dir)
    exp(model_name, data_name, base_path, )
    K.clear_session()


def exp(model_name, data_name, base_path):
    verbose = 0
    dau_name_arr = exp_utils.dau_name_arr
    mop_name_arr = exp_utils.mop_name_arr
    print(dau_name_arr)
    dau = get_dau(data_name)
    mop = Mop()
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    split_ratio = 0.5
    sample_num = 1
    model_path = model_conf.get_model_path(data_name, model_name)
    nb_classes = model_conf.fig_nb_classes

    ############################
    # exp
    ############################

    x_dau_test_arr, y_dau_test_arr, x_val_dict, y_val_dict = exp_utils.get_dau_data(x_test, y_test, dau, dau_name_arr,
                                                                                    ratio=split_ratio,
                                                                                    shuffle=True)

    for mop_name in mop_name_arr:
        for i in range(sample_num):
            df = None
            csv_path = os.path.join(base_path, "res_all_{}_{}.csv").format(mop_name, i)
            if mop_name == "ORI":
                x_select = np.concatenate(x_dau_test_arr, axis=0)
                y_select = np.concatenate(y_dau_test_arr, axis=0)
            else:
                x_select, y_select = exp_utils.get_mutation_data(mop_name, mop, x_dau_test_arr, y_dau_test_arr,
                                                                 data_name, seed=i, )
                x_s = x_select[y_select != -1]
                y_s = y_select[y_select != -1]
                temp_model_path = model_conf.get_temp_model_path(data_name, model_name, exp_name)
                imp_dict, retrain_time = retrain_detail(temp_model_path, x_s, y_s, x_train, y_train,
                                                        x_val_dict, y_val_dict,
                                                        model_path, nb_classes,
                                                        verbose=verbose, only_add=False)
                cov_trained_csv_data = get_retrain_csv_data("ALL", imp_dict, retrain_time)
                df = add_df(df, cov_trained_csv_data)
                df.to_csv(csv_path, index=False)


                del x_s
            del x_select


if __name__ == '__main__':
    exp_name = "exp_mutation"  # exp_repeat
    base_dir = "result_raw/ALL"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    for data_name, v_arr in tqdm(model_conf.model_data.items()):
        for model_name in v_arr:
            exec(model_name, data_name, base_dir)
