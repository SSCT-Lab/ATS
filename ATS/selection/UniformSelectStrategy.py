import bisect
import os
from collections import defaultdict

from ATS.selection.AbsSelectStrategy import AbsSelectStrategy
from utils.utils import get_data_by_label_with_idx
import numpy as np
import pandas as pd


# @deprecated
class UniformSelectStrategy(AbsSelectStrategy):

    def assert_Tx(self, Tx_i, target_size):
        if Tx_i.size == 0:
            raise ValueError("该lable下没有数据")
        if len(Tx_i) < target_size:
            raise ValueError("该lable下的数据不够选")

    def assert_S_mid(self, S_mid, target_size, symbol, idx_up, idx_mid):
        select_lable = None
        select_idx = []
        s_mid_len = len(S_mid)
        idx_mid = list(idx_mid)
        if s_mid_len == target_size:
            select_lable = symbol.SELECT_ALL_MID_LABLES
            select_idx = idx_mid
        elif s_mid_len < target_size:
            select_lable = symbol.SELECT_ALL_MID_LABLES_WITH_HIGH
            select_idx += idx_mid
            diff_size = target_size - s_mid_len
            idx_up_ran = list(np.random.randint(len(idx_up), size=diff_size))
            select_idx += idx_up_ran
            if len(select_idx) != target_size:
                raise ValueError("do not slect target_size points from S_up ")
        elif s_mid_len == 0:
            select_lable = symbol.SELECT_ZERO_LABLES
            diff_size = target_size - s_mid_len
            idx_up_ran = list(np.random.randint(len(idx_up), size=diff_size))
            select_idx += idx_up_ran
            raise ValueError("taget region does not have points")
        return select_lable, select_idx

    def datasets_select(self, Tx, Ty, n, M, ori_target_size, extra_size=0, base_path=None, is_analyze=False):
        print(ori_target_size)
        df = None
        csv_path = base_path + "/data_select_{}.csv".format(ori_target_size)
        csv_data = {}
        c_arr = []
        max_size_arr = []
        Xr_select = []
        select_lable_arr = []
        symbol = select_status()
        for i in range(n):
            if extra_size == 0:
                target_size = ori_target_size
            else:
                target_size = ori_target_size + 1
                extra_size -= 1
            C_select_i_map = None
            max_cov_point_size = -1
            print("i", i)
            csv_data["label"] = i
            Tx_i, Ty_i, T_idx_arr = get_data_by_label_with_idx(Tx, Ty, i)

            self.assert_Tx(Tx_i, target_size)
            Tx_prob_matrixc = M.predict(Tx_i)
            if len(Tx_i) == target_size:
                select_lable = symbol.SELECT_ALL_LABLES
                abs_idx = T_idx_arr
                C_select_i_map = self.get_ck_list_map(Tx_prob_matrixc, n, i)
            else:
                # S_mid
                ##########################################
                S_up, rel_idx_up, S_mid, rel_idx_mid, S_low, rel_idx_low \
                    = self.cluster_test_step.split_data_region_with_idx(Tx_prob_matrixc, i,
                                                                        np.array(range(len(Tx_prob_matrixc))))

                print("len(S0_i)", len(S_mid))
                select_lable, rel_select_idx = self.assert_S_mid(S_mid, target_size, symbol, rel_idx_up, rel_idx_mid)
                if select_lable is None:
                    select_lable, rel_select_idx, C_select_i_map, max_cov_point_size \
                        = self.datasets_select_detail(symbol, S_mid, n, i, target_size, rel_idx_mid,
                                                      base_path=base_path,
                                                      is_analyze=is_analyze)
                if C_select_i_map is None:
                    C_select_i_map = self.get_ck_list_map(Tx_prob_matrixc[rel_select_idx], n, i)
                abs_idx = T_idx_arr[rel_select_idx]
            s_pq_arr = self.pattern_fitness_step.get_cov_length_map(C_select_i_map, n, i, )
            s, c_i = self.pattern_fitness_step.get_cov_s_and_c(s_pq_arr, n)
            print("覆盖率 ", c_i)
            Xr_select.append(abs_idx)
            select_lable_arr.append(select_lable)
            csv_data["select_lable"] = select_lable
            c_arr.append(c_i)
            csv_data["div"] = c_i
            max_size_arr.append(max_cov_point_size)
            csv_data["max_cov"] = max_cov_point_size
            if df is None:
                df = pd.DataFrame(csv_data, index=[0])
            else:
                df.loc[df.shape[0]] = csv_data
            df.to_csv(csv_path, index=False)
        return Xr_select, select_lable_arr, c_arr, max_size_arr

    def datasets_select_detail(self, symbol, S_mid, n, i, target_size, idx_mid, base_path=None, is_analyze=False):
        temp_c_arr = []
        max_cov_point_size = target_size
        rel_mid_idx = []
        C_select_i_map = defaultdict(list)
        X_no_i = []
        select_lable = symbol.SELECT_MID_LABLES_CAM
        ck_list_map = self.get_ck_list_map(S_mid, n, i)
        C_Xr_list = list(map(list, zip(*ck_list_map.values())))
        if len(idx_mid) != len(C_Xr_list):
            raise ValueError("len idx not eq len data")
            ### step1
        all_rank_arr = []
        for ii, xi in enumerate(C_Xr_list):
            l = np.array(xi)[:, 4].sum()
            all_rank_arr.append([ii, l])
        # step2.
        all_rank_arr.sort(key=lambda x: (x[1]), reverse=False)
        # step3.
        all_rank_idx = np.array(all_rank_arr)[:, 0].astype("int").tolist()
        all_rank_cov_len = np.array(all_rank_arr)[:, 1].tolist()
        # if is_analyze:
        #     df = pd.DataFrame({"idx": all_rank_idx, "len": all_rank_cov_len})
        #     df.to_csv(base_path + "/{}_data_rank.csv".format(i))
        #     return
        # # step4.
        # C_Xr_list = np.array(C_Xr_list)[all_rank_idx]
        # # step5.
        # rel_mid_idx.append(all_rank_idx[0])
        # X_no_i.append(all_rank_idx[0])
        while len(rel_mid_idx) < target_size:
            max_s_i = 0
            max_idx = 0
            max_s_diff = 0
            max_Cr_select_i_map = None
            s_c_select = self.pattern_fitness_step.get_cov_pair_map_len(C_select_i_map, n, i)
            # coverage
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
                # step 3
                # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^2", time.time())
                Cx_union = self.pattern_fitness_step.statistic_union_map(Cx_insert, n, i)
                # print(Cx_union, "===")
                # step 3
                # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^3", time.time())
                s_c_union = self.pattern_fitness_step.get_cov_pair_map_len(Cx_union, n, i)
                # print(s_c_union, "===")
                # step 4
                s_diff = s_c_union - s_c_select  #
                if abs(s_diff) <= 0.001:
                    X_no_i.append(j)
                elif s_c_union > s_c_select:
                    if s_c_union > max_s_i:
                        max_s_i = s_c_union
                        max_idx = j
                        max_Cr_select_i_map = Cx_union
                        max_s_diff = s_diff
                    # step5
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
                X_no_i.append(max_idx)
                C_select_i_map = max_Cr_select_i_map.copy()
                all_rank_idx = all_rank_idx_copy.copy()
                all_rank_cov_len = all_rank_cov_len_copy.copy()
                if max_s_diff < 0.005:
                    pass
            else:
                if len(X_no_i) != len(S_mid):
                    raise ValueError("no point left but X_no_i does not hav all data points")
                if len(rel_mid_idx) == target_size:
                    select_lable = symbol.SELECT_MID_LABLES_CAM_ALL
                    max_cov_point_size = len(rel_mid_idx)
                    break
                else:
                    add_num = target_size - len(rel_mid_idx)
                    select_lable = symbol.SELECT_MID_LABLES_CAM_CTM
                    max_cov_point_size = len(rel_mid_idx)
                    print("cam max ={}", format(len(rel_mid_idx)))
                    Xr_ctm_idx = list(set(X_no_i) - set(rel_mid_idx))
                    C_Xr_CTM_list = np.array(C_Xr_list)[Xr_ctm_idx]
                    sorted_arr = []
                    for i_pq, x_pq in zip(Xr_ctm_idx, C_Xr_CTM_list):
                        len_s = np.array(x_pq)[:, 4].sum()
                        sorted_arr.append([i_pq, len_s])
                    sorted_arr.sort(key=lambda x: (x[1]), reverse=True)

                    select_ctm_x = np.array(sorted_arr)[:, 0].astype("int").tolist()
                    select_ctm_x = select_ctm_x[:add_num]
                    rel_mid_idx += select_ctm_x
                    break
        if is_analyze:
            csv_dir = base_path + "/data_select_profile/"
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = csv_dir + "{}.csv".format(i)
            df = pd.DataFrame(temp_c_arr)
            df.to_csv(csv_path)
        idx_mid = np.array(idx_mid)
        Xr_select_i = idx_mid[rel_mid_idx]
        # np.save(base_path + "/mid_rel_idx_{}.npy".format(i), S_mid[rel_mid_idx[-1]])
        # np.save(base_path + "/{}.npy".format(i), rel_mid_idx)
        if len(Xr_select_i) != target_size:
            raise ValueError("data points size not eq target_size")
        if len(Xr_select_i) != len(set(Xr_select_i)):
            raise ValueError("some data points  repeatly select")
        return select_lable, Xr_select_i, C_select_i_map, max_cov_point_size


class select_status(object):

    def __init__(self) -> None:
        self.SELECT_ALL_LABLES = 0  # all data
        self.SELECT_ZERO_LABLES = 1  # all data  but Smid no data
        self.SELECT_ALL_MID_LABLES = 2  # Smid all data sleceted
        self.SELECT_ALL_MID_LABLES_WITH_HIGH = 3  # extend data with high coff
        self.SELECT_MID_LABLES_CAM = 4  # data not enough use cam
        self.SELECT_MID_LABLES_CAM_ALL = 5  # data get max cov
        self.SELECT_MID_LABLES_CAM_CTM = 6  # data not enough use cam and ctm
