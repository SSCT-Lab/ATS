import os

from tqdm import tqdm

from utils import model_conf


def init_data():
    init_svhn_data()


def init_svhn_data():
    tran_path = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_path = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    # extra_path = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
    train_data_path = "./data/svhn/SVHN_train_32x32.mat"
    test_data_path = "./data/svhn/SVHN_test_32x32.mat"
    # extra_data_path = "./data/svhn/SVHN_extra_32x32.mat"
    if not os.path.exists(train_data_path):
        os.system("curl -o {} {}".format(train_data_path, tran_path))
    if not os.path.exists(test_data_path):
        os.system("curl -o {} {}".format(test_data_path, test_path))
    # if not os.path.exists(extra_data_path):
    #     os.system("curl -o {} {}".format(extra_data_path, extra_path))


def init_model():
    from gen_data import DauUtils
    from gen_model.gen_model_utils import gen_model
    for data_name, model_name_arr in tqdm(model_conf.model_data.items()):
        for model_name in model_name_arr:
            dau = DauUtils.get_dau(data_name)
            (x_train, y_train), (x_test, y_test) = dau.load_data(use_norm=True)
            gen_model(data_name, model_name, x_train, y_train, x_test, y_test)


def init_dirs():
    if not os.path.exists("./model"):
        os.makedirs("./model")
    if not os.path.exists("./data/svhn"):
        os.makedirs("./data/svhn")
    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result_raw"):
        os.makedirs("./result_raw")
    if not os.path.exists("./dau"):
        os.makedirs("./dau")
    if not os.path.exists("./temp_model"):
        os.makedirs("./temp_model")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    init_dirs()
    init_data()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # init_model()
