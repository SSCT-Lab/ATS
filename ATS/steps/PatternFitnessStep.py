import numpy as np

from ATS import ats_config
from ATS.steps.Utils import ATSUtils


class PatternFitnessStep(object):

    def __init__(self):
        self.ats_utils = ATSUtils()


    # put cov pair in SelectMap\
    def union_cov_maps(self, Cx, Cr_select_i_map, X_no_i):
        res_map = {}
        for x_pq in Cx:
            insert_falg = True
            [p, q, a_ins, b_ins, _] = x_pq
            CK_pq = Cr_select_i_map["{}_{}".format(p, q)].copy()
            if len(CK_pq) == 0:
                res_map["{}_{}".format(p, q)] = CK_pq
            else:
                for ix in range(len(CK_pq)):
                    [_, _, a_cur, b_cur, _] = CK_pq[ix]
                    # find  if lef gt right
                    if a_ins > a_cur:
                        continue
                    # the first lt
                    else:
                        CK_pq.insert(ix, x_pq)
                        insert_falg = False
                        break
            if insert_falg:
                CK_pq.append(x_pq)
            res_map["{}_{}".format(p, q)] = CK_pq
        return res_map

    def get_cov_length_map(self, ck_list_map, n, i, ):
        s_pq_arr = []
        pq_list = self.ats_utils.get_p_q_list(n, i)
        for (p, q) in pq_list:
            ck_list = ck_list_map["{}_{}".format(p, q)]
            Ck_pq = self.statistic_union(p, q, ck_list)
            s_pq = self.get_cov_length(Ck_pq)
            s_pq_arr.append(s_pq)
        return s_pq_arr

    # cal ipq union
    def statistic_union(self, p, q, Ck_pq_temp, sort=True):
        if len(Ck_pq_temp) == 0:
            return 0
        Ck_pq = Ck_pq_temp.copy()
        if sort:
            Ck_pq.sort(key=lambda x: (x[2]))
        res = []
        s_pre = Ck_pq[0][2]  # front line start
        e_pre = Ck_pq[0][3]  # front line end
        for i in range(1, len(Ck_pq)):
            s_cur = Ck_pq[i][2]  # cur line start
            e_cur = Ck_pq[i][3]  # cur line end
            if s_cur <= e_pre:
                # merge line
                e_pre = max(e_cur, e_pre)  # update length
            else:
                res.append([p, q, s_pre, e_pre, e_pre - s_pre])  # add result
                s_pre = s_cur
                e_pre = e_cur
        res.append([p, q, s_pre, e_pre, e_pre - s_pre])
        return res

    def get_cov_length(self, Ck_pq):
        total_length = 0
        for i in range(len(Ck_pq)):
            total_length += Ck_pq[i][3] - Ck_pq[i][2]
        return total_length

    def get_cov_pair_map_len(self, Ck_pq_map, n, i):
        l_total = 0
        for (p, q) in self.ats_utils.get_p_q_list(n, i):
            CK_pq = Ck_pq_map["{}_{}".format(p, q)]
            if len(CK_pq) != 0:
                l = np.array(CK_pq)[:, 4].sum()
                l_total += l
        return l_total

    def statistic_union_map(self, Ck_pq_map, n, i):
        res_map = {}
        for (p, q) in self.ats_utils.get_p_q_list(n, i):
            key = "{}_{}".format(p, q)
            CK_pq = Ck_pq_map[key]
            CK_pq = self.statistic_union(p, q, CK_pq, sort=False)
            res_map[key] = CK_pq
        return res_map

    def get_cov_c(self, s, n):
        c = s / ((1 - ats_config.boundary) * (self.get_cn2(n)))
        return c

    def get_cn2(self, n):
        return (1 / 2 * (n - 1) * (n - 2))

    def get_cov_s_and_c(self, s_pq_arr, n):
        s = np.array(s_pq_arr).sum()
        c = self.get_cov_c(s, n)
        return s, c
