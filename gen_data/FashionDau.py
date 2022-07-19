import os

from keras.datasets import fashion_mnist

from gen_data.Dau import Dau


class FashionDau(Dau):
    def __init__(self):
        super().__init__("fashion")
        self.train_size = 60000
        self.test_size = 10000
        self.nb_classes = 10

    # 用于扩增图片
    # 加载原始数据
    def load_data(self, use_norm=False):  # use_norm=False 用于图片扩增  # use_norm=True 用于训练数据和实验
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        if use_norm:
            x_test = x_test.astype('float32')
            x_train = x_train.astype('float32')
            x_train /= 255
            x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    # 获得当前的所有操作和参数
    def get_dau_params(self):
        params = {
            "SF": [(0.01, 0.13), (0.01, 0.13)],
            "RT": (5, 20),  # rotation
            "ZM": ((0.7, 1.5), (0.7, 1.5)),  # zoom
            "BR": 0.5,
            "SR": [10, 30],  # sheer
            "BL": "hard",  # blur
            "CT": [0.5, 1.5],
        }
        return params


if __name__ == '__main__':
    dau = FashionDau()
    dau.run("test")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # from utils import model_conf
    #
    # #
    # model_path = model_conf.get_model_path(model_conf.fashion, model_conf.LeNet1)
    # dau.get_acc_by_op(model_path)
    # model_path = model_conf.get_model_path(model_conf.fashion, model_conf.resNet20)
    # dau.get_acc_by_op(model_path)
