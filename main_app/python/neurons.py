
from tqdm import tqdm
from random import randint, shuffle, seed
import matplotlib.pyplot as plt
import numpy as np


axon_myelin_sheath = 6  # 髓鞘 长度
dendrite_length = 3  # 树突长度
cell_thred_max = 0.45  # 设置阈值和最大输入值比例
cell_thred_attenuate = 0.15  # 衰减系数 0.05需要衰减10次才能到最大值


def encode_data(d):
    result = []
    for e in d:
        if e == 0:
            result.append([ 0, 0, 1, 1])
        else:
            result.append([ 0, 0, 0, 0])
    result = np.array(result).T
    return result.tolist()


# 创建树突和轴突 axon 开启为树突 否则为轴突
def create_spine(number, axon=True):
    count = randint(number, number + int(number ** 0.5))
    spine = []
    for index in range(count):
        spine.append([0] * randint(1, dendrite_length))
    if axon:
        spine.append([0] * axon_myelin_sheath)
    return spine


# 创建抑制神经元
def create_inhibitory_list(number, pro):
    p_list = [1] * (number - int(number * pro))
    n_list = [-1] * int(number * pro)
    p_list.extend(n_list)
    shuffle(p_list)
    return p_list


# dendrites 树突个数
# axon 轴突个数
# inhib树突抑制比
class Neurons:
    def __init__(self, dendrites, axon, inhib):
        self.dendrites = create_spine(dendrites,axon=False)
        self._inhib = create_inhibitory_list(len(self.dendrites), inhib)
        self.axon = create_spine(axon,axon=True)
        self._thred_m = int(len(self.dendrites) * cell_thred_max * round(1 - inhib))
        self._thred = self._thred_m
        self._voltage = 0

    def input(self, d):
        for index in range(len(self.dendrites)):
            self.dendrites[index][0] = d[index] * self._inhib[index]  # 清除最左边值

    def output(self):
        d = []
        for index in range(len(self.axon)):
            d.append(self.axon[index][-1])
        return d

    def tick(self):
        self._voltage = 0
        for index in range(len(self.dendrites)):
            self._voltage = self._voltage + self.dendrites[index][-1]
            self.dendrites[index].insert(0, self.dendrites[index].pop())  # 向右移动一格
            self.dendrites[index][0] = 0  # 清除最左边值
        for index in range(len(self.axon) - 1):
            self.axon[index].insert(0, self.axon[index].pop())  # 向右移动一格
            self.axon[index][0] = self.axon[-1][-1]  # 设置值
        self.axon[-1].insert(0, self.axon[-1].pop())  # 向右移动一格
        if self._voltage >= self._thred:
            self._thred = self._thred + len(self.dendrites)
            self.axon[-1][0] = 1
        else:
            self.axon[-1][0] = 0
            self._thred = self._thred - self._thred * cell_thred_attenuate
            if self._thred <= self._thred_m:
                self._thred = self._thred_m


# cell_num 神经元个数
# n_input 输入的数据个数
# n_output 输出的数据个数
# dendrites 单个神经元内部树突个数
# axon 单个神经元内部树突个数
# inhib 突触抑制比例
class Networks:
    def __init__(self, cell_num, n_input, n_output, dendrites, axon, inhib):
        self.cell_num = cell_num
        self.output_num = n_output
        self.input_num = n_input
        self.loop_num = 0
        self.eva_active = [0] * cell_num
        self.dendrites_list = []
        self.axon_list = []
        self.dendrites_num = 0
        self.axon_num = 0
        self.cell = []
        for index in range(self.cell_num):
            _cell = Neurons(dendrites, axon, inhib)
            self.cell.append(_cell)
            self.dendrites_num = self.dendrites_num + len(_cell.dendrites)
            self.axon_num = self.axon_num + len(_cell.axon) - 1
        self.dendrites_list = list(range(self.dendrites_num))
        self.axon_list = list(range(self.axon_num))
        shuffle(self.dendrites_list)
        shuffle(self.axon_list)
        self.loop_num = min(self.dendrites_num - n_input, self.axon_num - n_output)
        self.dendrites_data = [0] * self.dendrites_num
        self.axon_data = [0] * self.axon_num

    def tick(self):
        self.self_transport()
        self.self_evaluate()
        for index in range(self.cell_num):
            self.cell[index].tick()

    def self_transport(self):
        # 提取所有神经元输出的数据
        count = 0
        for index in range(self.cell_num):
            for j in range(len(self.cell[index].axon) - 1):
                self.axon_data[self.axon_list.index(count)] = self.cell[index].axon[j][-1]
                count = count + 1
        # 把输出的数据通过loop_list传递给输出缓冲区
        for e, d in zip(self.dendrites_list[self.input_num:self.input_num + self.loop_num],
                        self.axon_data[self.output_num:self.output_num + self.loop_num]):
            self.dendrites_data[e] = d
        # 把所有输出值设置到输入中
        count = 0
        for index in range(self.cell_num):
            for j in range(len(self.cell[index].dendrites)):
                self.cell[index].dendrites[j][0] = self.dendrites_data[count] * self.cell[index]._inhib[j]
                count = count + 1

    def output(self):
        return self.axon_data[0:self.output_num]

    def input(self, d):
        for index in range(self.input_num):
            self.dendrites_data[self.dendrites_list[index]] = d[index]

    def self_evaluate(self):
        for index in range(self.cell_num):
            self.eva_active[index] = self.eva_active[index] + self.cell[index].axon[-1][-1]

    def clear_evaluate(self):
        self.eva_active = [0] * self.cell_num

    def update_thred_m(self, coe):
        self.clear_evaluate()
        e = encode_data([0] * self.input_num)
        for _c in tqdm(range(30)):
            for k in range(len(e)):
                self.input(e[k])
                self.tick()
        d = np.array(self.eva_active)
        d = d / d.max()
        self.clear_evaluate()
        for _i, _eum in enumerate(d):
            self.cell[_i]._thred_m = self.cell[_i]._thred_m - (1.0 - _eum) * coe

if __name__ == '__main__':
    # seed(0)
    n = Networks(3, 10, 3, 6, 7, 0.2)
    # print(n.__dict__)
    a=[1]*10
    x = range(200)
    y = []
    for i in x:
        n.input(a)
        n.tick()
        y.append(sum(n.output()))
    plt.plot(x, y)
    # plt.plot(x, z)
    plt.show()

    # eva = training_c([1] * 784, net, 50)
    # plt.bar(range(len(eva)), eva)
    # plt.show()
    #
    # eva = training_c(np.load("./temp/mnist_test.npy", allow_pickle=True)[0][0], net, 50)
    # plt.bar(range(len(eva)), eva)
    # plt.show()
    #
    # eva = training_c([0] * 784, net, 50)
    # plt.bar(range(len(eva)), eva)
    # plt.show()