import time
import numpy as np
from keras.engine.saving import load_model
from selection_method.necov_method import metrics
from utils import model_conf


##############
# cal coverage
#############

class CovInit(object):
    def __init__(self, X_train, Y_train, params: dict):
        if "model_path" in params.keys():
            model_path = params["model_path"]
        else:
            model_path = model_conf.get_model_path(params["data_name"], params["model_name"])
        if "layer_config_name" in params.keys():
            layer_config_name = params["layer_config_name"]
        else:
            layer_config_name = params["model_name"]
        ori_model = load_model(model_path)
        input_layer, layers = get_layers(layer_config_name, ori_model)
        self.lsc_config = self.init_lsc_config(layer_config_name, ori_model)
        self.model_path = model_path
        self.model_name = layer_config_name
        self.input_layer = input_layer
        self.layers = layers
        self.X_train = X_train
        self.Y_train = Y_train

        self.nbc = None
        self.kmnc = None
        self.lsc_map = {}
        self.dsc = None

    def init_lsc_config(self, model_name, model):
        layer_conf = {model_conf.LeNet5: ("dense", 8, 2000, 1e-5),  # 8
                      model_conf.LeNet1: ("dense", 8, 2000, 1e-5),  # 8
                      model_conf.resNet20: ('conv', 17, 100, 1e-4),  # 17
                      model_conf.vgg16: ('conv', 7, 100, 1e-4),
                      }
        v = layer_conf[model_name]
        lsc_config = ((v[0], model.layers[v[1]].output), v[2], v[3])
        return lsc_config

    def get_lsc_configs(self):
        return self.lsc_config

    def get_input_layer(self):
        return self.input_layer

    def get_layers(self):
        return self.layers

    def get_nbc(self, std=0):
        nbc = self.nbc
        if nbc is None:
            nbc = metrics.nbc(self.X_train, self.input_layer, self.layers, std=std)
            self.nbc = nbc
        return nbc

    def get_kmnc(self, k_bins=1000, time_limit=43200, max_select_size=None):
        kmnc = self.kmnc
        if kmnc is None:
            kmnc = metrics.kmnc(self.X_train, self.input_layer, self.layers,
                                k_bins=k_bins, time_limit=time_limit, max_select_size=max_select_size)
            self.kmnc = kmnc
        return kmnc

    def get_lsc(self, k_bins=1000, index=-1, threshold=None, u=100):
        layers, ub, th = self.get_lsc_configs()
        # print(layers)
        if index in self.lsc_map.keys():
            lsc = self.lsc_map[index]
        else:
            lsc = metrics.LSC(self.X_train, self.Y_train, self.input_layer, [layers], k_bins=k_bins, u=ub,
                              threshold=th)
            self.lsc_map[index] = lsc
        return lsc

    def get_dsc(self, k_bins=1000, time_limit=3600, u=2, ):
        dsc = self.dsc
        if dsc is None:
            dsc = metrics.DSC(self.X_train, self.Y_train, self.input_layer, self.layers, k_bins=k_bins,
                              time_limit=time_limit, u=u)
            self.dsc = dsc
        return dsc


class CovRank(object):
    def __init__(self, cov_initer: CovInit, model_path, x_s, y_s):
        self.cov_initer = cov_initer
        self.model_path = model_path
        self.x_s = x_s
        self.y_s = y_s

    def get_layers(self):
        return self.cov_initer.input_layer, self.cov_initer.layers

    def load_ori_model(self):
        return load_model(self.model_path)

    def cal_deepgini(self, ):
        s = time.time()
        pred_test_prob = self.load_ori_model().predict(self.x_s)
        e = time.time()
        t_collection = e - s

        s = time.time()
        metrics = np.sum(pred_test_prob ** 2, axis=1)
        rank_lst = np.argsort(metrics)
        e = time.time()
        t_selection_cam = e - s
        return 1 - metrics, t_collection, None, None, rank_lst, t_selection_cam,

    def cal_nac_cov(self, t=0.75, only_ctm=False):
        input_layer, layers = self.get_layers()
        nac = metrics.nac(self.x_s, input_layer, layers, t=t)

        if only_ctm:
            rank_lst2 = nac.rank_2(self.x_s)
            return None, None, None, None, rank_lst2, None
        else:
            rate = nac.fit()
            rank_lst = nac.rank_fast(self.x_s)
            rank_lst2 = nac.rank_2(self.x_s)
            return rate, nac.t_collection, rank_lst, nac.t_cam, rank_lst2, None

    def cal_nbc_cov(self, std=0, only_ctm=False):  # 0 0.5 1
        nbc = self.cov_initer.get_nbc(std=std)
        if only_ctm:
            rank_lst2 = nbc.rank_2(self.x_s, use_lower=True)
            return None, None, None, None, rank_lst2, None
        else:
            rate = nbc.fit(self.x_s, use_lower=True)

            rank_lst = nbc.rank_fast(self.x_s, use_lower=True)

            rank_lst2 = nbc.rank_2(self.x_s, use_lower=True)

            return rate, nbc.t_collection, rank_lst, nbc.t_cam, rank_lst2, None

    def cal_snac_cov(self, std=0, only_ctm=False):  # 0 0.5 1
        snac = self.cov_initer.get_nbc(std=std)
        if only_ctm:
            rank_lst2 = snac.rank_2(self.x_s, use_lower=False)
            return None, None, None, None, rank_lst2, None
        else:
            rate = snac.fit(self.x_s, use_lower=False)
            rank_lst = snac.rank_fast(self.x_s, use_lower=False)
            rank_lst2 = snac.rank_2(self.x_s, use_lower=False)
            return rate, snac.t_collection, rank_lst, snac.t_cam, rank_lst2, None

    def cal_kmnc_cov(self, k_bins=1000, max_select_size=None, time_limit=3600):  # 1000
        kmnc = self.cov_initer.get_kmnc(k_bins=k_bins, time_limit=time_limit, max_select_size=max_select_size)
        rate = kmnc.fit(self.x_s, )
        # rank_lst = kmnc.rank_fast(self.x_s, )
        rank_lst = kmnc.rank_greedy(self.x_s, )
        rank_lst2 = None
        return rate, kmnc.t_collection, rank_lst, kmnc.t_cam, rank_lst2, None

    def cal_tknc_cov(self, k=1):  # 1,2,3
        input_layer, layers = self.get_layers()
        tknc = metrics.tknc(self.x_s, input_layer, layers, k=k)
        rate = tknc.fit(list(range(len(self.x_s))))
        rank_lst = tknc.rank(self.x_s, )
        rank_lst2 = None
        return rate, tknc.t_collection, rank_lst, tknc.t_cam, rank_lst2, None

    def cal_lsc_cov(self, k_bins=1000, u=100, index=-1):
        lsc = self.cov_initer.get_lsc(k_bins=k_bins, index=index, u=u)
        rate = lsc.fit(self.x_s, self.y_s)
        rank_lst = lsc.rank_fast()
        return rate, lsc.t_collection, rank_lst, lsc.t_cam, None, None

    def cal_dsc_cov(self, k_bins=1000, u=2, time_limit=3600):
        dsc = self.cov_initer.get_dsc(k_bins=k_bins, time_limit=time_limit, u=u, )
        rate = dsc.fit(self.x_s, self.y_s)
        rank_lst = dsc.rank_fast()
        return rate, dsc.t_collection, rank_lst, dsc.t_cam, None, None


def get_layers(model_name, model):
    input = model.layers[0].output
    lst = []
    for index, layer in enumerate(model.layers):
        if 'activation' in layer.name:
            lst.append(index)
    lst.append(len(model.layers) - 1)
    if model_name == model_conf.LeNet5:
        layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                  model.layers[8].output, model.layers[9].output, model.layers[10].output]
        layers = list(zip(4 * ['conv'] + 3 * ['dense'], layers))
    elif model_name == model_conf.LeNet1:
        layers = [model.layers[2].output, model.layers[3].output, model.layers[5].output, model.layers[6].output,
                  model.layers[8].output, ]
        layers = list(zip(4 * ['conv'] + 1 * ['dense'], layers))
    elif model_name == model_conf.resNet20:
        layers = []
        for index in lst:
            layers.append(model.layers[index].output)
        layers = list(zip(19 * ['conv'] + 1 * ['dense'], layers))
    elif model_name == model_conf.vgg16 or model_name == model_conf.MyVgg16:  # vgg16
        layers = []
        for i in range(1, 19):
            layers.append(model.layers[i].output)
        for i in range(20, 23):
            layers.append(model.layers[i].output)
        layers = list(zip(18 * ['conv'] + 3 * ['dense'], layers))
    elif model_name == model_conf.MyLeNet5:
        layers = [model.layers[2].output, model.layers[3].output,
                  model.layers[5].output, model.layers[6].output,
                  model.layers[8].output, model.layers[9].output,
                  model.layers[11].output, model.layers[12].output,
                  model.layers[14].output, model.layers[15].output,
                  model.layers[17].output, model.layers[18].output,
                  model.layers[20].output, model.layers[21].output, model.layers[22].output, ]
        layers = list(zip(12 * ['conv'] + 3 * ['dense'], layers))

    else:
        raise ValueError("model {} do not have coverage layers config info".format(model_name))
    return input, layers
