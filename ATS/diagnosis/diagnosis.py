import os

from ATS.steps.Steps import Steps
import numpy as np
import pandas as pd


####################
# cov point analyze
####################
# @deprecated
class ATSdiagnosis(Steps):

    def ck_pq_analyze(self, ck_list_map, n, i, base_path, S0_i):
        df = None
        csv_data = {}
        for (pp, qq) in self.ats_utils.get_p_q_list(n, i):
            ck_i_list = ck_list_map["{}_{}".format(pp, qq)]
            for idx in range(len(ck_i_list)):
                p, q, a, b, s = ck_i_list[idx]
                vec = S0_i[idx]
                csv_data["i"] = i
                csv_data["p"] = p
                csv_data["q"] = q
                csv_data["idx"] = idx
                csv_data["len"] = b - a
                csv_data["a"] = a
                csv_data["b"] = b
                csv_data["vec_i"] = vec[i]
                csv_data["vec_p"] = vec[p]
                csv_data["vec_q"] = vec[q]
                csv_data["vec_max"] = max(vec)
                csv_data["vec_gini"] = 1 - np.sum(vec ** 2)
                if df is None:
                    df = pd.DataFrame(csv_data, index=[0])
                else:
                    df.loc[df.shape[0]] = csv_data

        base_path = base_path + "/ck_point"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        csv_path = base_path + "/" + "{}_ck_point_norank.csv".format(i)
        df.to_csv(csv_path, index=False)
        csv_path = base_path + "/" + "{}_ck_point_analyze.csv".format(i)
        df = df.sort_values(axis=0, by=["len"], ascending=False)
        df.to_csv(csv_path, index=False)
