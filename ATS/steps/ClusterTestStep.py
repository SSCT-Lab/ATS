from ATS import ats_config


class ClusterTestStep(object):
    def split_data_region_with_idx(self, Tx_prob_matrixc, i, idx):
        Tx_i_prob_vec = Tx_prob_matrixc[:, i]

        S1_i = Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary]
        idx_1 = idx[Tx_i_prob_vec < ats_config.boundary]
        S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]
        idx_0 = idx[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]

        S2_i = Tx_prob_matrixc[(Tx_i_prob_vec > ats_config.up_boundary)]
        idx_2 = idx[(Tx_i_prob_vec > ats_config.up_boundary)]

        return S2_i, idx_2, S0_i, idx_0, S1_i, idx_1

    def split_data_region(self, Tx_prob_matrixc, i):
        Tx_i_prob_vec = Tx_prob_matrixc[:, i]

        S1_i = Tx_prob_matrixc[Tx_i_prob_vec < ats_config.boundary]
        S0_i = Tx_prob_matrixc[(Tx_i_prob_vec >= ats_config.boundary) & (Tx_i_prob_vec < ats_config.up_boundary)]

        S2_i = Tx_prob_matrixc[(Tx_i_prob_vec > ats_config.up_boundary)]
        return S2_i, S0_i, S1_i
