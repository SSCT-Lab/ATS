import os

from tqdm import tqdm

import exp_utils
from exp_utils import mk_exp_dir
from gen_data.DauUtils import get_dau
from mop.mutation_op import Mop
from selection_method.nc_cover import get_cov_initer
from selection_method.selection_utils import prepare_rank_ps
from utils.train_utils import retrain_detail, get_retrain_csv_data
from utils import model_conf
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')
from utils.utils import add_df
from keras import backend as K



def exec(model_name, data_name):
    print(model_name, data_name)
    base_path = mk_exp_dir(exp_name, data_name, model_name, base_dir)
    exp(model_name, data_name, base_path)
    K.clear_session()


def exp(model_name, data_name, base_path):
    is_prepare_ps = False
    verbose = 0
    rank_name_list = exp_utils.rank_name_list
    dau_name_arr = exp_utils.dau_name_arr
    mop_name_arr = exp_utils.mop_name_arr
    dau = get_dau(data_name)
    mop = Mop()
    (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
    cov_initer = get_cov_initer(x_train, y_train, data_name, model_name)
    ps_path = "{}/ps_data/".format(base_path)
    time_path = "{}/time/".format(base_path)
    os.makedirs(time_path, exist_ok=True)
    os.makedirs(ps_path, exist_ok=True)

    select_size_ratio_arr = [0.025, 0.05, 0.075, 0.1]  # 0.025, 0.05, 0.075,0.1
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

    select_size_arr = []
    len_data = len(np.concatenate(y_dau_test_arr, axis=0))
    for select_size_ratio in select_size_ratio_arr:
        select_size = int(len_data * select_size_ratio)
        select_size_arr.append(select_size)
    max_select_size = select_size_arr[-1]
    for mop_name in mop_name_arr:
        for i in range(sample_num):
            df = None
            csv_path = os.path.join(base_path, "res_{}_{}.csv").format(mop_name, i)
            if mop_name == "ORI":
                x_select = np.concatenate(x_dau_test_arr, axis=0)
                y_select = np.concatenate(y_dau_test_arr, axis=0)
            else:
                x_select, y_select = exp_utils.get_mutation_data(mop_name, mop, x_dau_test_arr, y_dau_test_arr,
                                                                 data_name, seed=i, )
            save_path = os.path.join(ps_path, "{}_" + "{}_{}.npy".format(mop_name, i))
            print("len(x_select)", len(x_select))
            if is_prepare_ps:
                ps_csv_path = os.path.join(time_path, "{}_{}.csv".format(mop_name, i))
                df_ps = prepare_rank_ps(rank_name_list, model_path, x_select, save_path, cov_initer, max_select_size)
                if df_ps is None:
                    ...
                else:
                    if os.path.exists(ps_csv_path):
                        ps_csv_path = os.path.join(time_path, "{}_{}2.csv".format(mop_name, i))
                    df_ps.to_csv(ps_csv_path)

            idx_data = {}
            for name in rank_name_list:
                fp = save_path.format(name)
                ix = np.load(fp)
                idx_data[name] = ix
            idx_div = idx_data["DeepDiv"]
            print("mop_name :{}".format(mop_name), "keys:", idx_data.keys(), "select_size_arr:", select_size_arr)
            for cur_size_ratio, cur_select_size in tqdm(zip(select_size_ratio_arr, select_size_arr)):
                for k, idx in tqdm(idx_data.items()):
                    select_size = cur_select_size
                    name = str(k)
                    x_s, y_s = x_select[idx][:select_size], y_select[idx][:select_size]
                    x_s = x_s[y_s != -1]
                    y_s = y_s[y_s != -1]
                    assert len(x_s) == len(y_s)
                    effect_len = len(x_s)
                    if len(x_s) == 0:
                        imp_dict = {}
                        retrain_time = 0
                        imp_dict["all"] = 0
                    else:
                        temp_model_path = model_conf.get_temp_model_path(data_name, model_name, exp_name)
                        imp_dict, retrain_time = retrain_detail(temp_model_path, x_s, y_s, x_train, y_train,
                                                                x_val_dict, y_val_dict,
                                                                model_path, nb_classes,
                                                                verbose=verbose, only_add=False)
                    cov_trained_csv_data = get_retrain_csv_data(name, imp_dict, retrain_time)
                    if k != "DeepDiv":
                        ix_rank = idx[:select_size]
                        idx_div_rank = idx_div[:select_size]
                        sim_rate = len(set(idx_div_rank) & set(ix_rank)) / len(idx_div_rank)
                    else:
                        sim_rate = 0
                    cov_trained_csv_data["sim_rate"] = sim_rate
                    cov_trained_csv_data["effect_len"] = effect_len
                    cov_trained_csv_data["cur_select_size"] = cur_select_size
                    cov_trained_csv_data["cur_size_ratio"] = cur_size_ratio
                    df = add_df(df, cov_trained_csv_data)
                    df.to_csv(csv_path, index=False)
                    del x_s
            del x_select


if __name__ == '__main__':
    exp_name = "exp_mutation"
    base_dir = "result_raw/Div{}".format("")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for data_name, v_arr in tqdm(model_conf.model_data.items()):
        for model_name in v_arr:
            exec(model_name, data_name)
