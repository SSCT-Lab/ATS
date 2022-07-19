import numpy as np
from tqdm import tqdm

np.random.seed(4)
import keras.backend as K
from collections import defaultdict


class CES_ranker(object):
    def __init__(self):
        ...

    def build_neuron_tables(self, model, x_test, divide, output):
        total_num = x_test.shape[0]
        # init dict and its input
        neuron_interval = defaultdict(np.array)
        neuron_proba = defaultdict(np.array)
        layer = model.layers[-3]
        # test_output = build_testoutput(model, x_test)
        # output = test_output
        lower_bound = np.min(output, axis=0)
        upper_bound = np.max(output, axis=0)

        for index in range(output.shape[-1]):
            # compute interval
            # temp = (upper_bound[index] - lower_bound[index]) * .25
            # let interval = 30
            interval = np.linspace(
                lower_bound[index], upper_bound[index], divide)
            neuron_interval[(layer.name, index)] = interval
            neuron_proba[(layer.name, index)] = self.output_to_interval(
                output[:, index], interval) / total_num

        # save
        # np.save('neuron_interval.npy', neuron_interval)
        # np.save('neuron_proba.npy', neuron_proba)
        return neuron_interval, neuron_proba

    def build_testoutput(self, model, x_test):
        input_tensor = model.input
        layer = model.layers[-3]
        # get this layer's output
        output = layer.output
        output_fun = K.function([input_tensor], [output])
        # print(output_fun)
        # output = output_fun([x_test])[0]

        N = 1000
        output = output_fun([x_test[0:N]])[0]
        # input_shape= x_test.shape[0]
        inputshape_N = int(x_test.shape[0] / N)
        for i in range(inputshape_N - 1):
            tmpoutput = output_fun([x_test[N + i * N:2 * N + i * N]])[0]
            # print(len(output))
            output = np.append(output, tmpoutput, axis=0)

        if inputshape_N * N != x_test.shape[0]:
            tmpoutput = output_fun([x_test[inputshape_N * N:x_test.shape[0]]])[0]
            output = np.append(output, tmpoutput, axis=0)
        # print(len(output[0]))
        # output=output[0]

        output = output.reshape(output.shape[0], -1)
        # print(output[0])
        test_output = output
        return test_output

    def neuron_entropy(self, model, neuron_interval, neuron_proba, sample_index, test_output):
        total_num = sample_index.shape[0]
        if (total_num == 0):
            return -1e3
        neuron_entropy = []
        layer = model.layers[-3]
        output = test_output
        output = output[sample_index, :]
        # get lower and upper bound of neuron output
        # lower_bound = np.min(output, axis=0)
        # upper_bound = np.max(output, axis=0)
        for index in range(output.shape[-1]):
            # compute interval
            # print('index:%d' % index)
            interval = neuron_interval[(layer.name, index)]
            bench_proba = neuron_proba[(layer.name, index)]
            test_proba = self.output_to_interval(
                output[:, index], interval) / total_num
            test_proba = np.clip(test_proba, 1e-10, 1 - 1e-10)
            log_proba = np.log(test_proba)
            temp_proba = bench_proba.copy()
            temp_proba[temp_proba < (.5 / total_num)] = 0
            entropy = np.sum(log_proba * temp_proba)
            neuron_entropy.append(entropy)
        return np.array(neuron_entropy)

    def coverage(self, entropy):
        return np.mean(entropy)

    def output_to_interval(self, output, interval):
        num = []
        for i in range(interval.shape[0] - 1):
            num.append(np.sum(np.logical_and(
                output > interval[i], output < interval[i + 1])))
        return np.array(num)

    def selectsample(self, model, x_test, delta, iterate, neuron_interval, neuron_proba, test_output):
        test = x_test
        # print(test)
        batch = delta

        max_index0 = np.random.choice(range(test.shape[0]), replace=False, size=30)
        for i in tqdm(range(iterate)):
            # print('i:%d' % i)
            arr = np.random.permutation(test.shape[0])
            max_iter = 30
            e = self.neuron_entropy(model, neuron_interval,
                                    neuron_proba, max_index0, test_output)
            cov = self.coverage(e)
            max_coverage = cov

            temp_cov = []
            index_list = []
            # select
            for j in range(max_iter):
                # print('j:%d' % j)
                # arr = np.random.permutation(test.shape[0])
                start = int(np.random.uniform(0, test.shape[0] - batch))
                # print(start)
                temp_index = np.append(max_index0, arr[start:start + batch])
                index_list.append(arr[start:start + batch])
                e = self.neuron_entropy(model, neuron_interval,
                                        neuron_proba, temp_index, test_output)
                new_coverage = self.coverage(e)
                temp_cov.append(new_coverage)

            max_coverage = np.max(temp_cov)
            cov_index = np.argmax(temp_cov)
            max_index = index_list[cov_index]
            if (max_coverage <= cov):
                max_index = np.random.choice(range(test.shape[0]), replace=False, size=delta)
            max_index0 = np.append(max_index0, max_index)
        return max_index0

    def conditional_sample(self, model, x_test, sample_size):
        delta = 5
        iterate = int((sample_size - 30) / delta)
        test_output = self.build_testoutput(model, x_test)
        neuron_interval, neuron_proba = self.build_neuron_tables(model, x_test, delta, test_output)
        index_list = self.selectsample(model, x_test, delta, iterate, neuron_interval, neuron_proba, test_output)
        return np.array(list(index_list))

    def run(self, model, x_test, sample_size):
        idx = self.conditional_sample(model, x_test, sample_size)
        return idx
