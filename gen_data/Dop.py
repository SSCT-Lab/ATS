import abc
import os
import numpy as np
from keras.engine.saving import load_model
from tqdm import tqdm

from gen_data.MyImageDataGenerator import MyImageDataGenerator
from keras.utils import np_utils
import cv2


# 具体扩增操作实现
# 对比度

# 固定数值的改变
# 传入的都是0-255图片#
class DauOperator(object):
    def __init__(self, fill_mode='nearest'):  # "constant"
        self.gen = MyImageDataGenerator(fill_mode=fill_mode)

    ##########################
    # new operator
    ###################################################################################################################
    # 1. 平移
    def my_shift(self, x, shift_range, seed=None):
        width_shift_range, height_shift_range = shift_range
        params = self.gen.my_get_transform_params(x.shape, width_shift_range=width_shift_range,
                                                  height_shift_range=height_shift_range, type="range", seed=seed)
        x = self.gen.my_transform(x, params)
        return x

    # 2. 旋转
    def my_rotation(self, x, rotation_range, seed=None):
        params = self.gen.my_get_transform_params(x.shape, rotation_range=rotation_range, type="range", seed=seed)
        x = self.gen.my_transform(x, params)
        return x

    # 3. 错切
    def my_shear(self, x, shear_range, seed=None):
        # from keras_preprocessing.image import ImageDataGenerator
        # gen_sheer = ImageDataGenerator(shear_range=0.9)
        # x = gen_sheer.random_transform(x)
        params = self.gen.my_get_transform_params(x.shape, shear_range=shear_range, type="range", seed=seed)
        x = self.gen.my_transform(x, params)
        return x

    # 4 放缩
    def my_zoom(self, x, zoom_range, seed=None):
        params = self.gen.my_get_transform_params(x.shape, zoom_range=zoom_range, type="range", seed=seed)
        x = self.gen.my_transform(x, params)
        return x

    # 5 亮度
    def my_brightness(self, x, max_delta, seed=None):
        import tensorflow as tf

        tf.enable_eager_execution()
        img = tf.image.random_brightness(x, max_delta=max_delta)
        return img.numpy()

    # # 5 亮度
    # def my_brightness(self, x, brightness_range, seed=None):
    #     params = self.gen.my_get_transform_params(x.shape, brightness_range=brightness_range, type="range", seed=seed)
    #     x = self.gen.my_transform(x, params)
    #     return x
    # 对比度
    def my_contrast(self, x, contrast_range, seed=None):
        import tensorflow as tf

        tf.enable_eager_execution()
        c_lower, c_upper = contrast_range[0], contrast_range[1]
        img = tf.image.random_contrast(x, c_lower, c_upper, seed=None)
        return img.numpy()

    # 6 模糊
    def my_blur(self, x, mode=None, seed=None):
        if mode == 'hard':
            op_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif mode == "easy":
            op_list = [0, 1, 4, 9]
        else:
            op_list = [3, 4, 8]  # 2, 3, 5, 6, 7, 8
        # print(blur_op)
        # op_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        shape = x.shape
        blur = []
        # blur_method_arr = ["GaussianBlur", "medianBlur", "blur"]
        #
        # np.random.seed(seed)
        # blur_method = np.random.permutation(blur_method_arr)[0]
        #
        # if blur_method=="GaussianBlur":
        #     blur_filter = [(3, 3), (5, 5), (7, 7)]
        #
        np.random.seed(seed)
        blur_op = np.random.permutation(op_list)[0]
        # print("blur_op", blur_op, seed)
        if blur_op == 0:
            blur = cv2.blur(x, (2, 2))
        if blur_op == 1:
            blur = cv2.blur(x, (3, 3))
        if blur_op == 2:
            blur = cv2.blur(x, (4, 4))
        if blur_op == 3:
            blur = cv2.blur(x, (5, 5))
        if blur_op == 4:
            blur = cv2.GaussianBlur(x, (3, 3), 0)
        if blur_op == 5:
            blur = cv2.GaussianBlur(x, (5, 5), 0)
        if blur_op == 6:
            blur = cv2.GaussianBlur(x, (7, 7), 0)
        if blur_op == 7:
            blur = cv2.medianBlur(x, 3)
        if blur_op == 8:
            blur = cv2.medianBlur(x, 5)
        if blur_op == 9:
            blur = cv2.bilateralFilter(x, 6, 50, 50)
            # blur = cv2.bilateralFilter(img, 9, 75, 75)
        return blur.reshape(shape)

    # 7 噪声
    def my_noise(self, x, mode=None, seed=None,noise_op=None):
        x = np.copy(x)
        if mode == "hard":  # mnsit
            var = 0.1
            amount = 0.1
            k = 0.5
        elif mode == "easy":
            var = 0.001
            amount = 0.01
            k = 0.05
        else:
            var = 0.01
            amount = 0.04
            k = 0.1
        # params_arr = [1]
        shape = x.shape
        # params_arr = [1, 2, 3]
        params_arr = [1, 2, 3]
        np.random.seed(seed)
        if noise_op is None:
            noise_op = np.random.permutation(params_arr)[0]
        # print("noise_op", noise_op, seed)
        if noise_op == 1:  # Gaussian-distributed additive noise.
            x = x / 255
            row, col, ch = x.shape
            mean = 0
            # var = 0.01  #
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = x + gauss
            noisy = np.clip(noisy, 0, 1.0)
            noisy = np.uint8(noisy * 255)
            return noisy.reshape(shape)
        elif noise_op == 2:  # Replaces random pixels with 0 or 1.
            s_vs_p = 0.5
            # amount = 0.04
            out = np.copy(x)
            # Salt mode
            num_salt = np.ceil(amount * x.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in x.shape]
            out[tuple(coords)] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * x.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in x.shape]
            out[tuple(coords)] = 0
            return out.reshape(shape)
        elif noise_op == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
            # k = 0.1
            row, col, ch = x.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = x + x * k * gauss
            noisy = np.clip(noisy, 0, 255)
            return noisy.astype(np.uint8).reshape(x.shape)

    def get_dau_func_and_values(self, k, v, seed=None):
        mul_arr = [True, False]
        np.random.seed(seed)
        mul1, mul2 = np.random.choice(mul_arr, 2, replace=True)

        if k in ["SF"]:
            if mul1:
                v[0] = (-1 * v[0][1], -1 * v[0][0])
            if mul2:
                v[1] = (-1 * v[1][1], -1 * v[1][0])
            # print(mul1, mul2)
            # print(v)
        elif k in ["RT", "SR"]:
            if mul1:
                v = (-1 * v[1], -1 * v[0])
        else:  # "ZM" # BR BL NS # 放缩没正负  亮度没正负  模糊没正负  随机噪声没正负
            pass

        func_map = {
            "SF": self.my_shift,  # shift
            "RT": self.my_rotation,  # rotation
            "ZM": self.my_zoom,  # zoom
            "BR": self.my_brightness,  # brightness
            "SR": self.my_shear,  # sheer
            "BL": self.my_blur,  # blur
            "NS": self.my_noise,  # noise
            # "CT": self.my_contrast
        }
        return func_map[k], v
