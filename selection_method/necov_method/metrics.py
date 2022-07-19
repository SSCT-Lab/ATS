# #!/usr/bin/env python2
# # -*- coding: utf-8 -*-
#
import time
from collections import defaultdict

import numpy as np
from keras import Model
from scipy import stats
from tqdm import tqdm


class MetricsTime(object):
    def __init__(self, t_collection=0, t_cam=0, t_ctm=0):
        self.t_collection = t_collection
        self.t_cam = t_cam
        self.t_ctm = t_ctm
        self.first_add_info_time = True
        self.use_batch_predict = False

    def get_t_collection(self):
        return self.t_collection

    def get_t_cam(self):
        return self.t_cam

    def get_t_ctm(self):
        return self.t_ctm

    def predict_by_batch(self, test, i):
        if self.use_batch_predict:
            len_test = len(test)
            arr = np.linspace(0, len_test, num=11)
            num_arr = [int(x) for x in arr]
            temp_arr = []
            for ix in range(len(num_arr) - 1):
                start_ix = num_arr[ix]
                end_ix = num_arr[ix + 1]
                temp = i.predict(test[start_ix:end_ix])
                temp.astype(np.float32)
                temp_arr.append(temp)
            temp_arr = np.concatenate(temp_arr, axis=0)
        else:
            temp_arr = i.predict(test)
        return temp_arr


## deep gauge
class kmnc(MetricsTime):
    def __init__(self, train, input, layers, k_bins=1000, max_select_size=None, time_limit=43200):
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.k_bins = k_bins
        self.lst = []
        self.upper = []
        self.lower = []
        index_lst = []

        self.time_limit = time_limit
        self.max_select_size = max_select_size

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.upper.append(np.max(temp, axis=0))
            self.lower.append(np.min(temp, axis=0))

        self.upper = np.concatenate(self.upper, axis=0)
        self.lower = np.concatenate(self.lower, axis=0)
        self.neuron_num = self.upper.shape[0]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, test):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        act_num = 0
        for index in range(len(self.upper)):
            bins = np.linspace(self.lower[index], self.upper[index], self.k_bins)
            act_num += len(np.unique(np.digitize(self.neuron_activate[:, index], bins)))
        return act_num / float(self.k_bins * self.neuron_num)

    def get_big_bins(self, test):
        s = time.time()
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        big_bins = np.zeros((len(test), self.neuron_num, self.k_bins + 1))
        for n_index, neuron_activate in tqdm(enumerate(self.neuron_activate)):
            for index in range(len(neuron_activate)):
                bins = np.linspace(self.lower[index], self.upper[index], self.k_bins)
                temp = np.digitize(neuron_activate[index], bins)
                big_bins[n_index][index][temp] = 1

        big_bins = big_bins.astype('int')
        e = time.time()
        if self.first_add_info_time:
            self.t_collection += e - s
            self.first_add_info_time = False
        return big_bins

    def rank_fast(self, test):
        big_bins = self.get_big_bins(test)
        start = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))
        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover_num = (big_bins[initial] > 0).sum()
        cover_last = big_bins[initial]
        while True:
            end = time.time()
            if self.max_select_size is not None and len(subset) == self.max_select_size:
                print("max_select_size", len(subset))
                self.t_cam = end - start
                return subset
            if end - start >= self.time_limit:
                print("=======================time limit=======================")
                return subset
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                temp1 = np.bitwise_or(cover_last, big_bins[index])
                now_cover_num = (temp1 > 0).sum()
                if now_cover_num > max_cover_num:
                    max_cover_num = now_cover_num
                    max_index = index
                    max_cover = temp1
                    flag = True
            cover_last = max_cover
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            subset.append(max_index)
            print(len(subset), end - start)
        self.t_cam = end - start
        return subset

    def rank_greedy(self, test):
        big_bins = self.get_big_bins(test)
        start = time.time()
        subset = []
        lst = list(range(len(test)))

        np.random.seed(0)

        initial = np.random.permutation(len(test))[0]
        subset.append(initial)

        max_cover_num = (big_bins[initial] > 0).sum()
        cover_last = big_bins[initial]

        for index in lst:
            if index == initial:
                continue
            temp1 = np.bitwise_or(cover_last, big_bins[index])
            now_cover_num = (temp1 > 0).sum()
            if now_cover_num > max_cover_num:
                max_cover_num = now_cover_num
                cover_last = temp1
                subset.append(index)  #
        end = time.time()
        self.t_cam = end - start

        return subset


class nbc(MetricsTime):
    def __init__(self, train, input, layers, std=0):
        super().__init__()
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.std = std
        self.lst = []
        self.upper = []
        self.lower = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = self.predict_by_batch(train, i).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(train, i, l.shape[-1])
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.upper.append(np.max(temp, axis=0) + std * np.std(temp, axis=0))
            self.lower.append(np.min(temp, axis=0) - std * np.std(temp, axis=0))
        self.upper = np.concatenate(self.upper, axis=0)
        self.lower = np.concatenate(self.lower, axis=0)
        self.neuron_num = self.upper.shape[0]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, test, use_lower=False):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)

        act_num = 0
        act_num += (np.sum(self.neuron_activate > self.upper, axis=0) > 0).sum()
        if use_lower:
            act_num += (np.sum(self.neuron_activate < self.lower, axis=0) > 0).sum()

        if use_lower:
            return act_num / (2 * float(self.neuron_num))
        else:
            return act_num / float(self.neuron_num)

    def get_lower_and_upper_flag(self, test, use_lower):
        s = time.time()
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)

        upper = (self.neuron_activate > self.upper)
        if use_lower:
            lower = (self.neuron_activate < self.lower)
        else:
            lower = []

        e = time.time()
        if self.first_add_info_time:
            self.t_collection += e - s
            self.first_add_info_time = False
        return upper, lower

    def rank_fast(self, test, use_lower=False):
        upper, lower = self.get_lower_and_upper_flag(test, use_lower)
        s = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))

        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover_num = np.sum(upper[initial])
        if use_lower:
            max_cover_num += np.sum(lower[initial])
        cover_last_1 = upper[initial]
        if use_lower:
            cover_last_2 = lower[initial]
        while True:
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                temp1 = np.bitwise_or(cover_last_1, upper[index])
                cover1 = np.sum(temp1)
                if use_lower:
                    temp2 = np.bitwise_or(cover_last_2, lower[index])
                    cover1 += np.sum(temp2)
                if cover1 > max_cover_num:
                    max_cover_num = cover1
                    max_index = index
                    flag = True
                    max_cover1 = temp1
                    if use_lower:
                        max_cover2 = temp2
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            # lst.remove(max_index)
            subset.append(max_index)
            cover_last_1 = max_cover1
            if use_lower:
                cover_last_2 = max_cover2
            # print(max_cover_num)
        e = time.time()
        self.t_cam = e - s
        return subset

    def rank_2(self, test, use_lower=False):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        if use_lower:
            s = time.time()
            res = np.argsort(
                np.sum(self.neuron_activate > self.upper, axis=1) + np.sum(self.neuron_activate < self.lower, axis=1))[
                  ::-1]
            e = time.time()
        else:
            s = time.time()
            res = np.argsort(np.sum(self.neuron_activate > self.upper, axis=1))[::-1]
            e = time.time()
        self.t_ctm = e - s
        return res


class tknc(MetricsTime):
    def __init__(self, test, input, layers, k=2):
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = test
        self.input = input
        self.layers = layers
        self.k = k
        self.lst = []
        self.neuron_activate = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = self.predict_by_batch(test, i).reshape(len(test), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, i, l.shape[-1])
            if index == 'dense':
                temp = i.predict(test).reshape(len(test), l.shape[-1])
            self.neuron_activate.append(temp)
        self.neuron_num = np.concatenate(self.neuron_activate, axis=1).shape[-1]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self, choice_index):
        neuron_activate = 0
        for neu in self.neuron_activate:
            temp = neu[choice_index]
            neuron_activate += len(np.unique(np.argsort(temp, axis=1)[:, -self.k:]))
        return neuron_activate / float(self.neuron_num)

    def get_top_k_neuron(self):
        s = time.time()
        neuron = []
        layers_num = 0
        for neu in self.neuron_activate:
            neuron.append(np.argsort(neu, axis=1)[:, -self.k:] + layers_num)
            layers_num += neu.shape[-1]
        neuron = np.concatenate(neuron, axis=1)
        e = time.time()
        self.t_collection += e - s
        return neuron

    def rank(self, test):
        neuron = self.get_top_k_neuron()

        s = time.time()
        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))

        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover = len(np.unique(neuron[initial]))

        cover_now = neuron[initial]

        while True:
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                temp = np.union1d(cover_now, neuron[index])
                cover1 = len(temp)
                if cover1 > max_cover:
                    max_cover = cover1
                    max_index = index
                    flag = True
                    max_cover_now = temp
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            subset.append(max_index)
            cover_now = max_cover_now
        e = time.time()
        self.t_cam = e - s
        return subset


## deepxplore
class nac(MetricsTime):
    def __init__(self, test, input, layers, t=0):
        super().__init__()
        s = time.time()
        self.train = test
        self.input = input
        self.layers = layers
        self.t = t
        self.lst = []
        self.neuron_activate = []
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                # temp = i.predict(test).reshape(len(test), -1, l.shape[-1])
                temp = self.predict_by_batch(test, i).reshape(len(test), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, i, l.shape[-1])
            if index == 'dense':
                temp = i.predict(test).reshape(len(test), l.shape[-1])
            temp = 1 / (1 + np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_num = np.concatenate(self.neuron_activate, axis=1).shape[-1]
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def fit(self):
        neuron_activate = 0
        for neu in self.neuron_activate:
            neuron_activate += np.sum(np.sum(neu > self.t, axis=0) > 0)
        return neuron_activate / float(self.neuron_num)

    def get_upper(self, test):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            temp = 1 / (1 + np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)
        s = time.time()
        upper = (self.neuron_activate > self.t)
        e = time.time()
        self.t_collection += e - s
        return upper

    def rank_fast(self, test):
        upper = self.get_upper(test)
        s = time.time()

        subset = []
        no_idx_arr = []
        lst = list(range(len(test)))
        initial = np.random.choice(range(len(test)))

        # lst.remove(initial)
        no_idx_arr.append(initial)
        subset.append(initial)
        max_cover_num = np.sum(upper[initial])
        cover_last_1 = upper[initial]
        while True:
            flag = False
            for index in lst:
                if index in no_idx_arr:
                    continue
                temp1 = np.bitwise_or(cover_last_1, upper[index])
                cover1 = np.sum(temp1)
                if cover1 > max_cover_num:
                    max_cover_num = cover1
                    max_index = index
                    flag = True
                    max_cover1 = temp1
            if not flag or len(lst) == 1:
                break
            no_idx_arr.append(max_index)
            subset.append(max_index)
            cover_last_1 = max_cover1
            # print(max_cover_num)
        e = time.time()
        self.t_cam = e - s
        return subset

    def rank_2(self, test):
        self.neuron_activate = []
        for index, l in self.lst:
            if index == 'conv':
                temp = self.predict_by_batch(test, l).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
                # self.predict_by_batch2(test, l, l.output.shape[-1])
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            temp = 1 / (1 + np.exp(-temp))
            self.neuron_activate.append(temp.copy())
        self.neuron_activate = np.concatenate(self.neuron_activate, axis=1)

        s = time.time()
        res = np.argsort(np.sum(self.neuron_activate > self.t, axis=1))[::-1]
        e = time.time()
        self.t_ctm = e - s
        return res


## deepgini
def deep_metric(pred_test_prob):
    metrics = np.sum(pred_test_prob ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    return rank_lst


def deep_metric2(pred_test_prob):
    metrics = np.sum(pred_test_prob ** 2, axis=1)  #
    rank_lst = np.argsort(metrics)
    return rank_lst, 1 - metrics


# Surprise Adequacy
class LSC(MetricsTime):
    def __init__(self, train, label, input, layers, u=2000, k_bins=1000, threshold=None):

        """
        """
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.lst = []
        self.neuron_activate_train = []
        self.u = u
        self.k_bins = k_bins
        self.threshold = threshold
        self.test_score = []
        self.train_label = np.array(label)
        index_lst = []

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            temp = None
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.neuron_activate_train.append(temp.copy())  # len(train), l.shape[-1]
        self.neuron_activate_train = np.concatenate(self.neuron_activate_train, axis=1)  #
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

        self.class_matrix = None
        self._init_class_matrix()
        self.kdes, self.removed_cols = self._init_kdes()

    def _init_class_matrix(self):
        class_matrix = {}
        for i, lb in enumerate(self.train_label):
            if lb not in class_matrix:
                class_matrix[lb] = []
            class_matrix[lb].append(i)
        self.class_matrix = class_matrix

    def _init_kdes(self):
        class_matrix = self.class_matrix
        train_ats = self.neuron_activate_train
        num_classes = np.unique(self.train_label)
        removed_cols = []
        if self.threshold is not None:
            for lb in num_classes:
                col_vectors = np.transpose(train_ats[class_matrix[lb]])
                for i in range(col_vectors.shape[0]):
                    if (
                            np.var(col_vectors[i]) < self.threshold
                            and i not in removed_cols
                    ):
                        removed_cols.append(i)

        kdes = {}
        for lb in num_classes:
            refined_ats = np.transpose(train_ats[class_matrix[lb]])
            refined_ats = np.delete(refined_ats, removed_cols, axis=0)
            if refined_ats.shape[0] == 0:
                print("warning....  remove all")
                continue
            kdes[lb] = stats.gaussian_kde(refined_ats)
        return kdes, removed_cols

    def _get_lsa(self, kde, at, removed_cols):
        refined_at = np.delete(at, removed_cols, axis=0)
        return np.asscalar(-kde.logpdf(np.transpose(refined_at)))

    def fit(self, test, label):
        s = time.time()
        # print("LSC fit")
        self.neuron_activate_test = []
        self.test_score = []
        for index, l in self.lst:
            temp = None
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test = np.concatenate(self.neuron_activate_test, axis=1)  # 10000 10

        class_matrix = self._init_class_matrix()

        kdes, removed_cols = self.kdes, self.removed_cols

        for test_sample, label_sample in tqdm(zip(self.neuron_activate_test, label)):
            if label_sample in kdes.keys():
                kde = kdes[label_sample]
                self.test_score.append(self._get_lsa(kde, test_sample, removed_cols))
            else:
                self.test_score.append(0)
        e = time.time()
        self.t_collection += e - s
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        x = np.unique(np.digitize(self.test_score, bins))
        rate = len(np.unique(x)) / float(self.k_bins)
        return rate

    def get_sore(self):
        return self.test_score

    def get_u(self):
        return self.u

    def rank_fast(self):
        s = time.time()
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        score_bin = np.digitize(self.test_score, bins)
        score_bin_uni = np.unique(score_bin)
        res_idx_arr = []
        for x in score_bin_uni:
            np.random.seed(41)
            idx_arr = np.argwhere(score_bin == x).flatten()
            idx = np.random.choice(idx_arr)
            res_idx_arr.append(idx)
        # print(len(res_idx_arr), self.k_bins * self.get_rate())
        e = time.time()
        self.t_cam = e - s
        return res_idx_arr

    def rank_2(self):
        s = time.time()
        res = np.argsort(self.get_sore())[::-1]
        e = time.time()
        self.t_ctm = e - s
        return res


## DSC
class DSC(MetricsTime):
    def __init__(self, train, label, input, layers, u=2, k_bins=1000, threshold=10 ** -5, time_limit=3600):
        super().__init__(t_ctm=None)
        s = time.time()
        self.train = train
        self.input = input
        self.layers = layers
        self.lst = []
        self.std_lst = []
        self.mask = []
        self.neuron_activate_train = []
        index_lst = []
        self.u = u
        self.k_bins = k_bins
        self.threshold = threshold
        self.test_score = []

        self.time_limit = time_limit

        for index, l in layers:
            self.lst.append(Model(inputs=input, outputs=l))
            index_lst.append(index)
            i = Model(inputs=input, outputs=l)
            if index == 'conv':
                temp = i.predict(train).reshape(len(train), -1, l.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = i.predict(train).reshape(len(train), l.shape[-1])
            self.neuron_activate_train.append(temp.copy())
        self.neuron_activate_train = np.concatenate(self.neuron_activate_train, axis=1)
        self.train_label = np.array(label)
        self.lst = list(zip(index_lst, self.lst))
        e = time.time()
        self.t_collection += e - s

    def find_closest_at(self, at, train_ats):
        dist = np.linalg.norm(at - train_ats, axis=1)
        return (min(dist), train_ats[np.argmin(dist)])

    def fit(self, test, label):
        s = time.time()
        start = time.time()
        self.neuron_activate_test = []
        self.test_score = []
        for index, l in self.lst:
            if index == 'conv':
                temp = l.predict(test).reshape(len(test), -1, l.output.shape[-1])
                temp = np.mean(temp, axis=1)
            if index == 'dense':
                temp = l.predict(test).reshape(len(test), l.output.shape[-1])
            self.neuron_activate_test.append(temp.copy())
        self.neuron_activate_test = np.concatenate(self.neuron_activate_test, axis=1)

        class_matrix = {}
        all_idx = []
        for i, lb in enumerate(self.train_label):
            if lb not in class_matrix:
                class_matrix[lb] = []
            class_matrix[lb].append(i)
            all_idx.append(i)

        for test_sample, label_sample in tqdm(zip(self.neuron_activate_test, label)):
            end = time.time()
            if end - start >= self.time_limit:
                print("=======================time limit=======================")
                return None
            x = self.neuron_activate_train[class_matrix[label_sample]]
            a_dist, a_dot = self.find_closest_at(test_sample, x)
            y = self.neuron_activate_train[list(set(all_idx) - set(class_matrix[label_sample]))]
            b_dist, _ = self.find_closest_at(
                a_dot, y
            )
            self.test_score.append(a_dist / b_dist)

        e = time.time()
        self.t_collection += e - s
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        x = np.unique(np.digitize(self.test_score, bins))
        rate = len(np.unique(x)) / float(self.k_bins)
        return rate

    def get_sore(self):
        return self.test_score

    def get_u(self):
        return self.u

    def rank_2(self):
        s = time.time()
        res = np.argsort(self.get_sore())[::-1]
        e = time.time()
        self.t_ctm = e - s
        return res

    def rank_fast(self):
        s = time.time()
        bins = np.linspace(np.amin(self.test_score), self.u, self.k_bins)
        score_bin = np.digitize(self.test_score, bins)
        score_bin_uni = np.unique(score_bin)
        res_idx_arr = []
        for x in score_bin_uni:
            np.random.seed(41)
            idx_arr = np.argwhere(score_bin == x).flatten()
            idx = np.random.choice(idx_arr)
            res_idx_arr.append(idx)
        e = time.time()
        self.t_cam = e - s
        return res_idx_arr


if __name__ == '__main__':
    pass
