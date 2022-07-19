import abc
import os
import numpy as np
from keras.engine.saving import load_model
from tqdm import tqdm

from keras.utils import np_utils


class Dau(object):

    def __init__(self, data_name, nb_classes=10, ):
        self.base_dir = "dau"
        self.dau_dir = os.path.join(self.base_dir, data_name)
        self.params = self.get_dau_params()
        self.dau_op = None  # "constant"
        self.nb_classes = nb_classes
        self.cache_map = {}


    def load_dop(self):
        from gen_data.Dop import DauOperator
        self.dau_op = DauOperator()

    @abc.abstractmethod
    def load_data(self, use_norm=False):
        pass


    def init_dir(self):
        if not os.path.exists(self.dau_dir):
            os.makedirs(self.dau_dir)

    def dau_datasets(self, num=1):
        self.run("train", num=num)
        self.run("test", num=num)

    def get_dau_params(self):
        return self.get_base_dau_params()

    @staticmethod
    def get_base_dau_params():
        params = {
            "SF": [(0, 0.15), (0, 0.15)],
            "RT": (0, 15),  # rotation
            "ZM": ((0.8, 1.2), (0.8, 1.2)),  # zoom
            "BR": 0.2,
            "SR": [5, 20],  # sheer
            "BL": None,  # blur
            "CT": [0.5, 1.5],
        }
        print(params)
        return params

    def get_dau_ops(self):
        op_list = []
        for k, v in self.get_dau_params().items():
            op = self.get_op_abbr(k)
            op_list.append(op)
        return op_list

    def get_op_abbr(self, k):
        return k

    def get_data_save_path(self):
        return os.path.join(self.dau_dir, "{}_{}_{}.npy")  # e.g. dau/mnist WL_train_x.npy

    def run(self, prefix, num=1):
        self.load_dop()
        self.init_dir()
        (x_train, y_train), (x_test, y_test) = self.load_data(use_norm=False)
        params = self.get_dau_params()
        # for i in range(num):
        for k, v in params.items():
            img_list = []
            label_list = []
            dau_op_name = k
            print(k)
            x_path = self.get_data_save_path().format(dau_op_name, prefix, "x")
            y_path = self.get_data_save_path().format(dau_op_name, prefix, "y")
            if prefix == "train":
                data = zip(x_train, y_train)
            else:
                data = zip(x_test, y_test)
            for i, (x, y) in tqdm(enumerate(data)):
                dau_func, dau_v = self.dau_op.get_dau_func_and_values(k, v, seed=None)
                img = dau_func(x, dau_v, seed=None)
                img_list.append(img)
                label_list.append(y)
            xs = np.array(img_list)
            ys = np.array(label_list)
            np.save(x_path, xs)
            np.save(y_path, ys)
            print(np.max(xs), xs.dtype)

    def show_test(self, is_mnist=True):
        self.load_dop()
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        (x_train, y_train), (x_test, y_test) = self.load_data(use_norm=False)
        params = self.get_dau_params()

        # for i in range(num):
        x, y = x_test[0], y_test[0]
        print('ori')
        if is_mnist:
            shape = (28, 28)
        else:
            shape = (32, 32, 3)

        plt.imshow(x.reshape(shape) / 255., cmap="gray")
        plt.savefig("temp_fig/{}.png".format("ori"))
        # plt.show()

        print(params)
        for k, v in params.items():  # 每种扩增算子单独扩增成1种
            dau_op_name = self.get_op_abbr(k)
            print(dau_op_name, k, v)
            dau_func, dau_v = self.dau_op.get_dau_func_and_values(k, v, seed=None)
            img = dau_func(x, dau_v, seed=None)
            # img /= 255.
            plt.imshow(img.reshape(shape) / 255., cmap="gray")
            # plt.show()
            plt.savefig("temp_fig/{}.png".format(dau_op_name))

            # xs = np.array(img_list)
            # ys = np.array(label_list)
            # np.save(x_path, xs)
            # np.save(y_path, ys)

    def get_acc_by_op(self, model_path, prefix="test", nb_classes=10, use_norm=True):
        res_map = {}
        model = load_model(model_path)
        params = self.get_dau_params()
        for k, v in params.items():  # 每种扩增算子单独扩增成1种
            dau_op_name = self.get_op_abbr(k)
            x, y = self.load_dau_data(dau_op_name, prefix=prefix, use_norm=use_norm)
            acc = model.evaluate(x, np_utils.to_categorical(y, nb_classes))[1]
            print(dau_op_name, acc)
            res_map[dau_op_name] = acc
        return res_map

    # 加载扩增数据
    def load_dau_data(self, dau_op_name, prefix="test", use_norm=True, use_cache=True):
        # 默认使用缓存
        if use_cache:
            key = dau_op_name + "_" + prefix + "_" + str(int(use_norm))
            if key not in self.cache_map.keys():
                x_path = self.get_data_save_path().format(dau_op_name, prefix, "x")
                y_path = self.get_data_save_path().format(dau_op_name, prefix, "y")
                x = np.load(x_path)
                y = np.load(y_path)
                if use_norm:
                    x = x.astype('float32') / 255
                self.cache_map[key + "_x"] = x
                self.cache_map[key + "_y"] = y
            else:
                return self.cache_map[key + "_x"], self.cache_map[key + "_y"]
        else:
            # 不使用缓存
            x_path = self.get_data_save_path().format(dau_op_name, prefix, "x")
            y_path = self.get_data_save_path().format(dau_op_name, prefix, "y")
            x = np.load(x_path)
            y = np.load(y_path)
            if use_norm:
                x = x.astype('float32') / 255
        return x, y

    def clear_cache(self):
        print("clear key:{}".format(self.cache_map.keys()))
        self.cache_map = {}

    # 评估
    def evaluate(self, model_path, prefix="test", nb_classes=10):
        model = load_model(model_path)
        (x_train, y_train), (x_test, y_test) = self.load_data(use_norm=True)
        if prefix == "test":
            acc = model.evaluate(x_test, np_utils.to_categorical(y_test, nb_classes))[1]
        else:
            acc = model.evaluate(x_train, np_utils.to_categorical(y_train, nb_classes))[1]
        return acc
