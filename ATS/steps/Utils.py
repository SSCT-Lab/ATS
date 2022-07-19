class ATSUtils(object):

    def get_p_q_list(self, n, i):
        num_list = list(range(n))
        num_list.remove(i)
        import itertools
        pq_list = []
        for pq in itertools.combinations(num_list, 2):
            pq_list.append(pq)
        return pq_list
