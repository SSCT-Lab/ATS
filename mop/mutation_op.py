import numpy as np

# RP 重复 RG 随机生成 IR 不相关数据 LB 标签不均衡 NP 噪声扰动
from utils.utils import shuffle_data, get_data_by_label


class Mop(object):
    def __init__(self):
        self.nb_classes = 10
        pass

    # LB 标签不均衡
    # 返回删除后的数据
    def label_bias(self, x_ori_select, y_ori_select, seed=0):
        ratio_arr = [0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        np.random.seed(seed)
        ratio_arr = np.random.permutation(ratio_arr)
        x_arr = []
        y_arr = []
        np.random.seed(seed)
        idx_arr = np.random.permutation(100000)
        seed_ix = 0
        for i in range(self.nb_classes):
            ratio = ratio_arr[i]
            x_i, y_i = get_data_by_label(x_ori_select, y_ori_select, i)
            x_i, y_i = shuffle_data(x_i, y_i, seed=idx_arr[seed_ix])
            num = int(ratio * len(x_i))
            x_temp, y_temp = x_i[:num], y_i[:num]
            x_arr.append(x_temp)
            y_arr.append(y_temp)
            seed_ix += 1
        x_ori_select = np.concatenate(x_arr, axis=0)
        y_ori_select = np.concatenate(y_arr, axis=0)
        return x_ori_select, y_ori_select

    # RG 随机生成  随机生成的图片数据
    # 以下都返回新增加的数据
    def synthetic_data(self, len_x_ori_select, img_shape, ratio=0.2, seed=0):
        num = int(ratio * len_x_ori_select)
        np.random.seed(seed)
        idx_arr = np.random.permutation(100000)[:num]
        x_arr = []
        for cur_seed in idx_arr:
            np.random.seed(cur_seed)
            choice = np.random.choice([0, 1, 2])
            if choice == 0:
                # 创建随机生成的像素点图
                np.random.seed(cur_seed)
                x_temp = np.random.rand(img_shape[0], img_shape[1], img_shape[2])
                x_arr.append(x_temp)
            elif choice == 1:
                # 创建随机生成的纯色图
                np.random.seed(cur_seed)
                v_arr = np.random.uniform(size=img_shape[2])
                x_temp = np.zeros(img_shape)
                for i in range(img_shape[2]):
                    v = v_arr[i]
                    x_temp_i = np.full((img_shape[0], img_shape[1]), v)
                    x_temp[:, :, i] = x_temp_i
                x_arr.append(x_temp)
            else:
                # 创建随机生成的条状图(横竖条)
                x_temp = np.zeros(img_shape)
                np.random.seed(cur_seed)
                seed_list = np.random.randint(0, 100000, size=img_shape[1])
                for i in range(img_shape[2]):
                    np.random.seed(seed_list[i])
                    vj_arr = np.random.uniform(size=img_shape[1])
                    x_temp_i = []
                    for j in range(img_shape[1]):
                        vj = vj_arr[j]
                        x_temp_i_j = np.array([vj] * img_shape[0])
                        x_temp_i.append(x_temp_i_j)
                    x_temp_i = np.array(x_temp_i)
                    np.random.seed(seed_list[i])
                    if np.random.choice([0, 1]):
                        x_temp_i = np.rot90(x_temp_i)
                    x_temp[:, :, i] = x_temp_i
                x_arr.append(x_temp)
        y_arr = np.array([-1] * len(x_arr))
        assert len(x_arr) == num
        return x_arr, y_arr

    # RP 随机挑选重复数据
    def repeat_data(self, x_ori_select, y_ori_select, ratio=0.5, seed=0):
        dup_num = int(len(x_ori_select) * ratio)
        x_ori_select, y_ori_select = shuffle_data(x_ori_select, y_ori_select, seed)
        x_temp, y_temp = x_ori_select[:dup_num], y_ori_select[:dup_num]
        assert len(x_temp) == dup_num
        return x_temp, y_temp

    # 不相关数据
    def irrelevant_data(self, len_x_ori_select, x_extra, ratio=0.2, seed=0):
        num = int(ratio * len_x_ori_select)
        np.random.seed(seed)
        x_extra = x_extra[np.random.permutation(len(x_extra))]
        x_temp = x_extra[:num]
        y_temp = np.array([-1] * len(x_temp))
        assert len(x_temp) == num
        return x_temp, y_temp

    # 敌意样本 adv
    def adv_data(self, len_x_ori_select, x_extra, y_extra, ratio=0.2, seed=0):
        num = int(ratio * len_x_ori_select)
        if num >= len(x_extra):
            raise ValueError(num, len(x_extra), "ADV 不够了")
        x_extra, y_extra = shuffle_data(x_extra, y_extra, seed=seed)
        x_temp = x_extra[:num]
        y_temp = y_extra[:num]
        assert len(x_temp) == num
        return x_temp, y_temp

    # 噪声 NP
    # ratio 噪声数据比例
    def noise_data(self, x_ori_select, y_ori_select, ratio=0.5, seed=1):
        dup_num = int(len(x_ori_select) * ratio)
        x_ori_select, y_ori_select = shuffle_data(x_ori_select, y_ori_select, seed)
        x_temp, y_temp = x_ori_select[:dup_num], y_ori_select[:dup_num]
        x_dup = []
        np.random.seed(seed)
        seed_list = np.random.randint(0, 100000, len(x_temp))
        for x, seed in zip(x_temp, seed_list):
            x2 = self.add_noise(x, seed)
            x_dup.append(x2)
        assert len(x_dup) == dup_num
        return np.array(x_dup), y_temp

    # 数据破损
    def corruption_data(self, x_ori_select, y_ori_select, ratio=0.5, seed=1):
        dup_num = int(len(x_ori_select) * ratio)
        x_ori_select, y_ori_select = shuffle_data(x_ori_select, y_ori_select, seed)
        x_temp, y_temp = x_ori_select[:dup_num], y_ori_select[:dup_num]
        x_cor = []
        np.random.seed(seed)
        seed_list = np.random.randint(0, 100000, len(x_temp))
        for x, seed in zip(x_temp, seed_list):
            x2 = self.add_corrup(x, seed)
            x_cor.append(x2)
        y_cor = np.array([-1] * len(x_temp))
        assert len(x_cor) == dup_num
        return np.array(x_cor), y_cor

    # 数据投毒
    def poison_data(self, x_ori_select, y_ori_select, ratio=0.5, seed=1):
        dup_num = int(len(x_ori_select) * ratio)
        x_ori_select, y_ori_select = shuffle_data(x_ori_select, y_ori_select, seed)
        x_temp, y_temp = x_ori_select[:dup_num], y_ori_select[:dup_num]
        x_cor = []
        np.random.seed(seed)
        seed_list = np.random.randint(0, 100000, len(x_temp))
        for x, seed in zip(x_temp, seed_list):
            x2 = self.add_poison(x, seed)
            x_cor.append(x2)
        y_cor = np.array([-1] * len(x_temp))
        assert len(x_cor) == dup_num
        return np.array(x_cor), y_cor

    # 投毒
    @staticmethod
    def add_poison(img, seed):
        x = img.copy()
        img_shape = x.shape
        corp_size = img_shape[0] // 4
        np.random.seed(seed)
        corp_position = np.random.randint(0, img_shape[0] - corp_size, size=2)
        np.random.seed(seed)
        x_poison = np.random.rand(corp_size, corp_size, img_shape[2])
        x[corp_position[0]:corp_position[0] + corp_size,
        corp_position[1]:corp_position[1] + corp_size,
        :] = x_poison
        return x

    # 破损
    @staticmethod
    def add_corrup(img, seed):
        x = img.copy()
        img_shape = x.shape
        corp_size = int(img_shape[0] // 1.5)
        np.random.seed(seed)
        corp_position = np.random.randint(0, img_shape[0] - corp_size, size=2)
        x[
        corp_position[0]:corp_position[0] + corp_size,
        corp_position[1]:corp_position[1] + corp_size,
        :] = 0
        return x

    # 噪声
    @staticmethod
    def add_noise(img, seed):
        x = img.copy()
        row, col, ch = x.shape
        mean = 0
        var = 0.001
        sigma = var ** 0.5
        np.random.seed(seed)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = x + gauss
        noisy = np.clip(noisy, 0, 1.0)
        return noisy.astype('float32').reshape(x.shape)
