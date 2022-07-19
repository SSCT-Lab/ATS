import functools
import os
import time
import numpy as np
from keras.engine.saving import load_model

from selection_method.necov_method.neural_cov import CovRank
from selection_method.ranker import Ranker

from utils.utils import add_df


def get_rank_func(rank_name_list, model_path, x_s, cov_initer, max_select_size, th=0.001):
    ori_model = load_model(model_path)
    ranker = Ranker(ori_model, x_s)
    func_list = []
    name_func_map = {
        "Gini": ranker.gini_rank,
        "MAX_P": ranker.max_p_rank,
        "NOISE": ranker.noise_rank,
        "DeepDiv": functools.partial(ranker.div_rank, th=th),
        "DeepDivCTM": ranker.div_rank_ctm,
        "CES": functools.partial(ranker.ces_rank, select_size=max_select_size),
        "Random": ranker.random_rank,
        "LSA": functools.partial(ranker.lsa_rank, cov_initer=cov_initer)
    }
    for name in rank_name_list:
        func_list.append(name_func_map[name])
    return func_list


def get_cov_func(cov_name_list, model_path, cov_initer, x_s, y_s):
    cov_ranker = CovRank(cov_initer, model_path, x_s, y_s)
    func_list = []
    name_func_map = {
        "NAC": cov_ranker.cal_nac_cov,
        "NBC": cov_ranker.cal_nbc_cov,
        "SNAC": cov_ranker.cal_snac_cov,
        "TKNC": cov_ranker.cal_tknc_cov,
    }
    for cov_name in cov_name_list:
        func_list.append(name_func_map[cov_name])
    return func_list


def prepare_rank_ps(rank_name_list, model_path, x_s, save_path, cov_initer, max_select_size):
    print("prepare ps ...")
    df = None
    func_list = get_rank_func(rank_name_list, model_path, x_s, cov_initer, max_select_size)
    for name, func in zip(rank_name_list, func_list):
        p = save_path.format(name)
        if os.path.exists(p):
            continue
        print(name)
        csv_data = {}
        csv_data["name"] = name
        s = time.time()
        rank_lst = func()
        # assert len(rank_lst) >= max_select_size
        e = time.time()
        csv_data["time"] = e - s
        np.save(p, rank_lst)
        df = add_df(df, csv_data)
    return df


def prepare_cov_ps(cov_name_list, model_path, x_s, save_path, cov_initer, max_select_size):
    print("prepare cov  ps ...")
    df = None
    func_list = get_cov_func(cov_name_list, model_path, x_s, cov_initer, max_select_size)
    for name, func in zip(cov_name_list, func_list):
        _, _, p, _, _, _ = save_path.format(name)
        if os.path.exists(p):
            continue
        print(name)
        csv_data = {}
        csv_data["name"] = name
        s = time.time()
        rank_lst = func()
        # assert len(rank_lst) >= max_select_size
        e = time.time()
        csv_data["time"] = e - s
        np.save(p, rank_lst)
        df = add_df(df, csv_data)
    return df
