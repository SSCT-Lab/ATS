import os

from utils import model_conf
from gen_model.model_cifar10_resNet20 import model_cifar_resNet20
from gen_model.model_cifar_vgg16 import model_cifar_vgg16
from gen_model.model_fashion import model_fashion_LeNet_1
from gen_model.model_fashion_resNet20 import model_fashion_resNet20
from gen_model.model_mnist import model_mnist_LeNet_5, model_mnist_LeNet_1
from gen_model.model_svhn import model_svhn_LeNet5
from gen_model.model_svhn_vgg16 import model_svhn_vgg16



# 实验2 模型   训练集不加数据
def gen_model(data_name, model_name, X_train, Y_train, X_test, Y_test):
    fp = model_conf.get_model_path(data_name, model_name)
    gen_model_detail(fp, data_name, model_name, X_train, Y_train, X_test, Y_test, )


# # 实验2模型  训练集 加扩增数据
# def gen_exp2_1_model(data_name, model_name, X_train, Y_train, X_test, Y_test, ):
#     fp = model_conf.get_exp2_1_model_path(data_name, model_name)
#     gen_model_detail(fp, model_name, X_train, Y_train, X_test, Y_test, )


def gen_model_detail(fp, data_name, model_name, X_train, Y_train, X_test, Y_test, ):
    if not os.path.exists(fp):  # 如果没有模型
        if model_name == model_conf.LeNet5 and data_name == model_conf.mnist:
            model_mnist_LeNet_5(X_train, Y_train, X_test, Y_test, fp)
        elif model_name == model_conf.LeNet1 and data_name == model_conf.mnist:
            model_mnist_LeNet_1(X_train, Y_train, X_test, Y_test, fp)
        elif model_name == model_conf.LeNet1 and data_name == model_conf.fashion:
            model_fashion_LeNet_1(X_train, Y_train, X_test, Y_test, fp)
        elif model_name == model_conf.resNet20 and data_name == model_conf.fashion:
            model_fashion_resNet20(X_train, Y_train, X_test, Y_test, fp)
        elif model_name == model_conf.LeNet5 and data_name == model_conf.svhn:
            model_svhn_LeNet5(X_train, Y_train, X_test, Y_test, fp)
        elif model_name == model_conf.vgg16 and data_name == model_conf.svhn:
            model_svhn_vgg16(X_train, Y_train, X_test, Y_test, fp)
        elif model_name == model_conf.resNet20 and data_name == model_conf.cifar10:
            model_cifar_resNet20(X_train, Y_train, X_test, Y_test, fp)
        elif model_name == model_conf.vgg16 and data_name == model_conf.cifar10:
            model_cifar_vgg16(X_train, Y_train, X_test, Y_test, fp)
        else:
            raise ValueError("no model")
