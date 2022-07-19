from utils import model_conf


def get_dau(data_name):
    from gen_data.CifarDau import CifarDau
    from gen_data.FashionDau import FashionDau
    from gen_data.MnistDau import MnistDau
    from gen_data.SvhnDau import SvhnDau
    if data_name == model_conf.mnist:
        return MnistDau()
    if data_name == model_conf.fashion:
        return FashionDau()
    if data_name == model_conf.svhn:
        return SvhnDau()
    if data_name == model_conf.cifar10:
        return CifarDau()


def get_data_size(data_name):
    if data_name == model_conf.mnist:
        train_size, test_size = 60000, 10000
    elif data_name == model_conf.fashion:
        train_size, test_size = 60000, 10000
    elif data_name == model_conf.svhn:
        train_size, test_size = 73257, 26032
    elif data_name == model_conf.cifar10:
        train_size, test_size = 50000, 10000
    else:
        raise ValueError()
    return train_size, test_size
