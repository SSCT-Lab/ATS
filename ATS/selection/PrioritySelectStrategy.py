import bisect
import os
import time
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from ATS.selection.AbsSelectStrategy import AbsSelectStrategy
import pandas as pd
from utils.utils import get_data_by_label_with_idx


class PrioritySelectStrategy(AbsSelectStrategy):

    def sort_ix_by_len(self, x_all, x_all_len):
        assert len(x_all) == len(x_all_len)
        dic = {}
        for ix, ix_len in zip(x_all, x_all_len):
            dic[ix] = ix_len
        assert len(dic) == len(x_all)

        sort_list_dict = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        sort_ix, sort_len = zip(*sort_list_dict)
        return sort_ix, sort_len

    def get_priority_sequence(self, Tx, Ty, n, M, base_path=None, prefix=None, is_save_ps=False, th=0.001):
        Xr_select, Xr_select_len, Xr_others, Xr_others_len, c_arr, max_size_arr, idx_others = \
            self.get_max_coverage_sequence(Tx,
                                           Ty,
                                           n,
                                           M,
                                           base_path=base_path,
                                           use_add=True,
                                           is_save_ps=is_save_ps,
                                           prefix=prefix,
                                           th=th, )

        x_all = np.concatenate(Xr_select, axis=0)
        x_all_len = np.concatenate(Xr_select_len, axis=0)
        sort_select_ix, sort_select_len = self.sort_ix_by_len(x_all, x_all_len)

        x_others = np.concatenate(Xr_others, axis=0)
        x_others_len = np.concatenate(Xr_others_len, axis=0)
        sort_others_ix, sort_others_len = self.sort_ix_by_len(x_others, x_others_len)

        np.random.seed(0)
        idx_others = np.concatenate(idx_others, axis=0)
        shuffle_ix = np.random.permutation(len(idx_others))
        idx_others = idx_others[shuffle_ix]

        sort_ix = np.concatenate([sort_select_ix, sort_others_ix, idx_others], axis=0)
        # assert len(sort_ix) == len(Ty)
        # assert len(sort_ix) == len(set(sort_ix))
        return sort_ix, c_arr, max_size_arr

    # no target_size
    def get_max_coverage_sequence(self, Tx, Ty, n, M, base_path=None, use_add=False, is_save_ps=False, prefix="",
                                  th=0.001):
        df = None
        csv_data = {}
        c_arr = []  # cov
        max_size_arr = []  # max_cov
        Xr_select = []  # select idx
        Xr_select_len = []  # select idx with cov length

        Xr_others = []  # left idx
        Xr_others_len = []  # left idx with cov length

        idx_others = []
        for i in tqdm(range(n)):
            csv_data["label"] = i
            Tx_i, Ty_i, T_idx_arr = get_data_by_label_with_idx(Tx, Ty, i)
            Tx_prob_matrixc = M.predict(Tx_i)
            S_up, rel_idx_up, S_mid, rel_idx_mid, S_low, rel_idx_low \
                = self.cluster_test_step.split_data_region_with_idx(Tx_prob_matrixc, i,
                                                                    np.array(range(len(Tx_prob_matrixc))))

            rel_select_idx, idx_max_diff_arr, rel_others_idx, ctm_max_diff_arr, C_select_i_map, max_cov_point_size = \
                self.get_priority_sequence_detail(
                    S_mid, n, i,
                    rel_idx_mid,
                    use_add=use_add,
                    th=th, )
            abs_ix_up = T_idx_arr[rel_idx_up]
            idx_others.append(abs_ix_up)
            abs_idx = T_idx_arr[rel_select_idx]
            abs_idx_others = T_idx_arr[rel_others_idx]
            # assert len(set(abs_idx) & set(abs_idx_others)) == 0
            # assert len(set(abs_ix_up) & set(abs_idx_others)) == 0
            # assert len(set(abs_idx) & set(abs_ix_up)) == 0
            Xr_select.append(abs_idx)
            Xr_select_len.append(idx_max_diff_arr)
            Xr_others.append(abs_idx_others)
            Xr_others_len.append(ctm_max_diff_arr)
            csv_data["len(S_up)"] = len(S_up)
            csv_data["len(S_mid)"] = len(S_mid)
            # cov
            s_pq_arr = self.pattern_fitness_step.get_cov_length_map(C_select_i_map, n, i, )
            s, c_i = self.pattern_fitness_step.get_cov_s_and_c(s_pq_arr, n)
            csv_data["div"] = c_i
            c_arr.append(c_i)

            # max_cov:
            max_size_arr.append(max_cov_point_size)
            csv_data["max_cov"] = max_cov_point_size
            if df is None:
                df = pd.DataFrame(csv_data, index=[0])
            else:
                df.loc[df.shape[0]] = csv_data
            if base_path is not None:
                csv_path = base_path + "/data_select.csv"
                df.to_csv(csv_path, index=False)
        if is_save_ps:
            ps_path = base_path + "/ps"
            ps_path_all = base_path + "/ps_all"
            os.makedirs(ps_path, exist_ok=True)
            os.makedirs(ps_path_all, exist_ok=True)
            for i in range(n):
                idx_arr = Xr_select[i]
                if prefix == "":
                    save_path = ps_path + "/{}.npy".format(i)
                else:
                    save_dir = ps_path + "/{}".format(prefix)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = save_dir + "/{}.npy".format(i)
                np.save(save_path, idx_arr)
            x_all_idx = np.concatenate(Xr_select, axis=0)
            save_path2 = ps_path_all + "/{}all.npy".format(prefix)
            np.save(save_path2, x_all_idx)
        return Xr_select, Xr_select_len, Xr_others, Xr_others_len, c_arr, max_size_arr, idx_others

    def get_priority_sequence_detail(self, S_mid, n, i, idx_mid, use_add=False, th=0.001, use_fine=False):
        temp_c_arr = []
        rel_mid_idx = []
        idx_max_diff_arr = []

        ctm_mid_idx = []
        ctm_max_diff_arr = []

        C_select_i_map = defaultdict(list)
        X_no_i = []
        ck_list_map = self.get_ck_list_map(S_mid, n, i)
        C_Xr_list = list(map(list, zip(*ck_list_map.values())))
        # if len(idx_mid) != len(C_Xr_list):
        #     raise ValueError("len idx not eq len data")
        all_rank_arr = []
        for ii, xi in enumerate(C_Xr_list):
            l = np.array(xi)[:, 4].sum()
            all_rank_arr.append([ii, l])
        all_rank_arr.sort(key=lambda x: (x[1]), reverse=False)
        all_rank_idx = np.array(all_rank_arr)[:, 0].astype("int").tolist()
        all_rank_cov_len = np.array(all_rank_arr)[:, 1].tolist()
        max_size = 0
        while True:
            max_s_i = 0
            max_idx = 0
            max_s_diff = 0
            max_Cr_select_i_map = None
            s_c_select = self.pattern_fitness_step.get_cov_pair_map_len(C_select_i_map, n, i)
            c = self.pattern_fitness_step.get_cov_c(s_c_select, n)
            temp_c_arr.append(c)
            # print("# current no select data point", len(X_no_i))
            all_rank_cov_len_copy = all_rank_cov_len.copy()
            all_rank_idx_copy = all_rank_idx.copy()
            for iix in range(len(all_rank_idx) - 1, -1, -1):
                j = all_rank_idx[iix]
                if j in X_no_i:
                    continue
                Cx = C_Xr_list[j]
                Cx_insert = self.pattern_fitness_step.union_cov_maps(Cx, C_select_i_map, X_no_i)
                Cx_union = self.pattern_fitness_step.statistic_union_map(Cx_insert, n, i)
                s_c_union = self.pattern_fitness_step.get_cov_pair_map_len(Cx_union, n, i)
                s_diff = s_c_union - s_c_select
                if abs(s_diff) <= th:
                    X_no_i.append(j)
                elif s_c_union > s_c_select:
                    if s_c_union > max_s_i:
                        max_s_i = s_c_union
                        max_idx = j
                        max_Cr_select_i_map = Cx_union
                        max_s_diff = s_diff
                    if iix != 0 and max_s_diff >= all_rank_cov_len[iix - 1]:
                        # print("selected lables: ", len(rel_mid_idx), "early stop in: ", iix, "cur max_s_diff", max_s_diff)
                        break
                    else:
                        all_rank_cov_len_copy.remove(all_rank_cov_len[iix])
                        all_rank_idx_copy.remove(j)
                        ins_idx = bisect.bisect(all_rank_cov_len_copy, s_diff)
                        all_rank_idx_copy.insert(ins_idx, j)
                        all_rank_cov_len_copy.insert(ins_idx, s_diff)
                        # print(iix, "--->", ins_idx)
                else:
                    X_no_i.append(j)
            if max_s_i != 0:
                rel_mid_idx.append(max_idx)
                idx_max_diff_arr.append(max_s_diff)
                X_no_i.append(max_idx)
                C_select_i_map = max_Cr_select_i_map.copy()
                all_rank_idx = all_rank_idx_copy.copy()
                all_rank_cov_len = all_rank_cov_len_copy.copy()
                if max_s_diff < 0.005:
                    pass
            else:
                max_size = len(rel_mid_idx)
                if use_add:
                    s = time.time()
                    Xr_ctm_idx = list(set(X_no_i) - set(rel_mid_idx))
                    for iix in all_rank_idx:
                        j = all_rank_idx[iix]
                        if j in Xr_ctm_idx:
                            ctm_mid_idx.append(j)
                            cov_len = all_rank_cov_len[j]
                            ctm_max_diff_arr.append(cov_len)
                    e = time.time()
                    break
                else:
                    break

        assert len(rel_mid_idx) == len(idx_max_diff_arr)
        idx_mid = np.array(idx_mid)
        Xr_select_i = idx_mid[rel_mid_idx]
        Xr_others_i = idx_mid[ctm_mid_idx]
        # if len(Xr_select_i) != len(set(Xr_select_i)):
        #     raise ValueError("some data points  repeatly select")
        # if len(Xr_others_i) != len(set(Xr_others_i)):
        #     raise ValueError("some data points  repeatly select")
        # assert len(set(Xr_select_i) & set(set(Xr_others_i))) == 0
        return Xr_select_i, idx_max_diff_arr, Xr_others_i, ctm_max_diff_arr, C_select_i_map, max_size
