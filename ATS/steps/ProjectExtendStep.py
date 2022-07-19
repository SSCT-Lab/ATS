import numpy as np

from ATS import ats_config


class ProjectExtendStep(object):

    def get_projection_matrixc(self, X, p, q, n, i):
        x_k_dot_matrixc = []
        for x_k in X:
            x_k_dot = self.get_projection_point(i, p, q, n, x_k)
            x_k_dot_matrixc.append(x_k_dot)
        return np.array(x_k_dot_matrixc)

    def get_projection_point(self, i, p, q, n, A):
        one_third = 1 / 3
        two_third = 2 / 3
        A_dot = np.zeros(A.shape)
        A_dot[i] = two_third * A[i] - one_third * A[p] - one_third * A[q] + one_third
        A_dot[p] = two_third * A[p] - one_third * A[q] - one_third * A[i] + one_third
        A_dot[p] = two_third * A[q] - one_third * A[p] - one_third * A[i] + one_third
        return A_dot

    # 2.extend
    def get_i_distance_list(self, X, i):
        i_distance_list = []
        for x_k_dot in X:
            d = ats_config.up_boundary - x_k_dot[i]
            i_distance_list.append(d)
        return i_distance_list

    def extend_line(self, X, i):
        x_k_dot_dot_matrixc = X.copy()
        n = len(x_k_dot_dot_matrixc[0])
        for x_k_dot in x_k_dot_dot_matrixc:
            d = 1 - x_k_dot[i]
            for j in range(n):
                if j == i:
                    x_k_dot[j] = ats_config.boundary
                    continue
                else:
                    x_k_dot[j] = ((1 - ats_config.boundary) / d) * x_k_dot[j]
        return x_k_dot_dot_matrixc

