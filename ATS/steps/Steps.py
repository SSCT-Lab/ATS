from ATS.steps.ClusterTestStep import ClusterTestStep
from ATS.steps.PatternFitnessStep import PatternFitnessStep
from ATS.steps.PatternGatherStep import PatternGatherStep
from ATS.steps.ProjectExtendStep import ProjectExtendStep
from ATS.steps.Utils import ATSUtils


class Steps(object):
    def __init__(self):
        self.cluster_test_step = ClusterTestStep()  # test cluster
        self.project_extend_step = ProjectExtendStep()  # project and extend
        self.pattern_gather_step = PatternGatherStep()  # fault pattern gather
        self.pattern_fitness_step = PatternFitnessStep()  # Fitness metric calculate
        self.ats_utils = ATSUtils()  # tools

    def get_ck_list_map(self, S0_i, n, i):
        pq_list = self.ats_utils.get_p_q_list(n, i)
        ck_map = {}
        for (p, q) in pq_list:
            S0_projection_matrixc = self.project_extend_step.get_projection_matrixc(S0_i, p, q, n, i)
            i_distance_list = self.project_extend_step.get_i_distance_list(S0_projection_matrixc, i)
            x_k_dot_dot_matrixc = self.project_extend_step.extend_line(S0_projection_matrixc, i)
            ck_i_list = self.pattern_gather_step.get_cov_pair(i_distance_list, x_k_dot_dot_matrixc, p, q)
            if len(ck_i_list) != 0:
                ck_map["{}_{}".format(p, q)] = ck_i_list
            if len(i_distance_list) == len(x_k_dot_dot_matrixc) == len(ck_i_list):
                pass
            else:
                raise ValueError("len ck list  not eq data size")
        return ck_map
