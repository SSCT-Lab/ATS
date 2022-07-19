import os

from keras.datasets import mnist

from gen_data.Dau import Dau


class MnistDau(Dau):
    def __init__(self):
        super().__init__("mnist")
        self.train_size = 60000
        self.test_size = 10000
        self.nb_classes = 10

    def load_data(self, use_norm=False):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
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

    def get_dau_params(self):
        params = {
            "SF": [(0, 0.15), (0, 0.15)],
            "RT": (0, 40),  # rotation
            "ZM": ((0.5, 2.0), (0.5, 2.0)),  # zoom
            "BR": 0.5,
            "SR": [10, 40.0],  # sheer
            "BL": "hard",  # blur
            "CT": [0.5, 1.5],
        }
        print(params)
        return params


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dau = MnistDau()
    dau.run("test")
    # dau.show_test()
    # from utils import model_conf
    #
    # model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet5)
    # dau.get_acc_by_op(model_path)
    # model_path = model_conf.get_model_path(model_conf.mnist, model_conf.LeNet1)
    # dau.get_acc_by_op(model_path)
