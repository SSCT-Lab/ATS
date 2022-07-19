import keras
import numpy as np
import pandas as pd


def get_data_by_label(X, Y, label, ):
    idx_arr = np.where(Y == label)
    y = Y[idx_arr]
    x = X[idx_arr]
    return x, y


def get_data_by_label3(X, Y, Z, label, ):
    idx_arr = np.where(Y == label)
    y = Y[idx_arr]
    x = X[idx_arr]
    z = Z[idx_arr]
    return x, y, z


# 带所选下标
def get_data_by_label_with_idx(X, Y, label, ):
    idx_arr = np.where(Y == label)
    y = Y[idx_arr]
    x = X[idx_arr]
    return x, y, idx_arr[0]


def remove_data_by_label(X, Y, label, ):
    idx_arr = np.where(Y != label)
    y = Y[idx_arr]
    x = X[idx_arr]
    return x, y


# 输出信息
def param2txt(file_path, msg, mode="w"):
    f = open(file_path, mode)
    f.write(msg)
    f.close


# 复制模型层
def model_copy(model):
    original_layers = [l for l in model.layers]
    new_model = keras.models.clone_model(model)
    for index, layer in enumerate(new_model.layers):
        original_layer = original_layers[index]
        original_weights = original_layer.get_weights()
        layer.set_weights(original_weights)
    return new_model


# 分割数据
def split_data_by_right(ori_model, x, y):
    prob_matrixc = ori_model.predict(x)
    ys_psedu = np.argmax(prob_matrixc, axis=1)  #
    x_right = x[y == ys_psedu]
    x_wrong = x[y != ys_psedu]
    y_right = y[y == ys_psedu]
    y_wrong = y[y != ys_psedu]
    return x_right, x_wrong, y_right, y_wrong


def split_data_by_right_with_psedu(ori_model, x, y):
    prob_matrixc = ori_model.predict(x)
    ys_psedu = np.argmax(prob_matrixc, axis=1)  #
    x_right = x[y == ys_psedu]
    x_wrong = x[y != ys_psedu]
    y_right = y[y == ys_psedu]
    y_wrong = y[y != ys_psedu]
    return x_right, x_wrong, y_right, y_wrong, ys_psedu


# 混洗数据
def shuffle_data(X, Y, seed=None):
    if len(X) != len(Y):
        raise ValueError("size X not eq Y")
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X, Y = X[shuffle_indices], Y[shuffle_indices]
    return X, Y


def shuffle_data3(X, Y, Z, seed=None):
    if len(X) != len(Y):
        raise ValueError("size X not eq Y")
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X, Y, Z = X[shuffle_indices], Y[shuffle_indices], Z[shuffle_indices]
    return X, Y, Z


# 添加一条数据
def add_df(df, csv_data):
    if df is None:  # 如果是空的
        df = pd.DataFrame(csv_data, index=[0])
    else:
        df.loc[df.shape[0]] = csv_data
    return df


# 小数点后两位
def num_to_str(num, trunc=2):
    return format(num, '.{}f'.format(trunc))


# 所有元素向前移动
def ahead_list(arr: list, num=1):
    temp_arr = arr.copy()
    for i in range(num):
        b = temp_arr.pop(0)
        temp_arr.append(b)
    return temp_arr
