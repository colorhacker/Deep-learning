
from random import randint, shuffle, seed
import matplotlib.pyplot as plt
import numpy as np


axon_myelin_sheath = 5  # 髓鞘 长度
dendrite_length = 3  # 树突长度
cell_thred_max = 0.5  # 设置阈值和最大输入值比例
cell_thred_attenuate = 0.05  # 衰减系数 0.05需要衰减10次才能到最大值


# 创建树突和轴突 axon 开启为树突 否则为轴突
def create_spine(number, axon=True):
    count = randint(number, number + int(number ** 0.5))
    spine = []
    for i in range(count):
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
        for i in range(len(self.dendrites)):
            self.dendrites[i][0] = d[i] * self._inhib[i]  # 清除最左边值

    def output(self):
        d = []
        for i in range(len(self.axon)):
            d.append(self.axon[i][-1])
        return d

    def tick(self):
        self._voltage = 0
        for i in range(len(self.dendrites)):
            self._voltage = self._voltage + self.dendrites[i][-1]
            self.dendrites[i].insert(0, self.dendrites[i].pop())  # 向右移动一格
            self.dendrites[i][0] = 0  # 清除最左边值
        for i in range(len(self.axon) - 1):
            self.axon[i].insert(0, self.axon[i].pop())  # 向右移动一格
            self.axon[i][0] = self.axon[-1][-1]  # 设置值
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
        self.dendrites_list = []
        self.axon_list = []
        self.dendrites_num = 0
        self.axon_num = 0
        self.cell = []
        for i in range(self.cell_num):
            _cell = Neurons(dendrites,axon, inhib)
            # _cell = Neurons(randint(dendrites, dendrites + int(dendrites ** 0.5)),
            #                 randint(axon, axon + int(axon ** 0.5)), inhib)
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
        for i in range(self.cell_num):
            self.cell[i].tick()

    def self_transport(self):
        # 提取所有神经元输出的数据
        count = 0
        for i in range(self.cell_num):
            for j in range(len(self.cell[i].axon) - 1):
                self.axon_data[self.axon_list.index(count)] = self.cell[i].axon[j][-1]
                count = count + 1
        # 把输出的数据通过loop_list传递给输出缓冲区
        for e, d in zip(self.dendrites_list[self.input_num:self.input_num + self.loop_num],
                        self.axon_data[self.output_num:self.output_num + self.loop_num]):
            self.dendrites_data[e] = d
        # 进行输入设置
        count = 0
        for i in range(self.cell_num):
            for j in range(len(self.cell[i].dendrites)):
                self.cell[i].dendrites[j][0] = self.dendrites_data[count] * self.cell[i]._inhib[j]
                count = count + 1

    def output(self):
        return self.axon_data[0:self.output_num]

    def input(self, d):
        for i in range(self.input_num):
            self.dendrites_data[self.dendrites_list[i]] = d[i]


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

