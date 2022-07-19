import os

from utils import model_conf
from gen_data.Dau import Dau
import numpy as np


class SvhnDau(Dau):
    def __init__(self):
        super().__init__("svhn")
        self.train_size = 73257
        self.test_size = 26032
        self.nb_classes = 10

    def load_data(self, use_norm=False):
        from utils import SVNH_DatasetUtil
        (x_train, y_train), (x_test, y_test) = SVNH_DatasetUtil.load_data()
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        print(np.max(x_test), np.min(x_test), x_test.dtype)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        y_test = np.argmax(y_test, axis=1)
        y_train = np.argmax(y_train, axis=1)
        return (x_train, y_train), (x_test, y_test)

    def get_dau_params(self):
        params = {
            "SF": [(0, 0.15), (0, 0.15)],
            "RT": (5, 20),  # rotation
            "ZM": ((0.8, 1.5), (0.8, 1.5)),  # zoom
            "BR": 0.3,
            "SR": [10, 30],  # sheer
            "CT": [0.5, 1.5],
            "BL": None,  # blur
        }

        return params


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dau = SvhnDau()
    dau.run("test")
