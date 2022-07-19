from ATS.measure.ATSmeasure import ATSmeasure
from ATS.selection.PrioritySelectStrategy import PrioritySelectStrategy


class ATS(object):
    def __init__(self):
        self.ats_method = PrioritySelectStrategy()  # 排序
        self.ats_measure = ATSmeasure()  # 度量

    # # 计算覆盖率和不同维度的覆盖方差
    # def cal_coverage_and_var(self, Tx, Ty, n, M, base_path=None, is_anlyze=False, suffix=""):
    #     return self.ats_measure.cal_d_v(Tx, Ty, n, M)

    # 获取优先级序列
    def get_priority_sequence(self, Tx, Ty, n, M, base_path=None,prefix=None, is_save_ps=False,
                             th=0.001):
        return self.ats_method.get_priority_sequence(Tx, Ty, n, M, base_path=base_path,  prefix=prefix,
                                                     is_save_ps=is_save_ps, th=th)

    # def get_ctm_priority_sequence(self):
    #     ...
