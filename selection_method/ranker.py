from ATS.ATS import ATS
from selection_method.necov_method import metrics
import numpy as np

from selection_method.rank_method.CES.condition import CES_ranker
from utils import model_conf


class Ranker(object):
    def __init__(self, model, x):
        self.model = model
        self.x = x

    def gini_rank(self):
        pred_test_prob = self.model.predict(self.x)
        gini_rank = metrics.deep_metric(pred_test_prob)
        return gini_rank

    def max_p_rank(self):
        pred_test_prob = self.model.predict(self.x)
        metrics = np.max(pred_test_prob, axis=1)
        max_p_rank = np.argsort(metrics)
        return max_p_rank

    def noise_rank(self):
        pred_test_prob = self.model.predict(self.x)
        matrix = np.sort(pred_test_prob, axis=1)[:, ::-1]
        pd_score_arr = []
        for p in matrix:
            sum = 0
            for i in range(len(p) - 1):
                diff = p[i] - p[i + 1]
                assert diff >= 0
                sum += diff
            pd_score = sum / len(p)
            pd_score_arr.append(pd_score)
        pd_score_rank = np.argsort(pd_score_arr)
        return pd_score_rank

    def div_rank(self, th=0.001):
        ats = ATS()
        pred_test_prob = self.model.predict(self.x)
        Y_psedu_select = np.argmax(pred_test_prob, axis=1)
        div_rank, _, _ = ats.get_priority_sequence(self.x, Y_psedu_select, model_conf.fig_nb_classes,
                                                   self.model, is_save_ps=False, th=th)
        return div_rank



    def ces_rank(self, select_size=None):
        if select_size is None:
            select_size = len(self.x)
        return CES_ranker().run(self.model, self.x, select_size)

    def lsa_rank(self, cov_initer, y_test=None):
        if y_test is None:
            pred_test_prob = self.model.predict(self.x)
            Y_psedu_select = np.argmax(pred_test_prob, axis=1)
            y_test = Y_psedu_select
        lsc = cov_initer.get_lsc()
        rate = lsc.fit(self.x, y_test)
        rank_lst = lsc.rank_2()
        return np.array(rank_lst)

    def random_rank(self):
        return np.random.permutation(len(self.x))



