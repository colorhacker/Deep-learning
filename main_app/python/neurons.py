
from random import randint, shuffle, seed
import matplotlib.pyplot as plt
import numpy as np


#创建树突和轴突 axon 开启为树突 否则为轴突
def create_spine(length,axon=True):
    count = randint(length, length + int(length ** 0.5))
    spine = []
    for i in range(count):
        spine.append([0] * randint(1, int(count ** 0.5)))
    if axon:
        spine.append([0] * length)
    return spine


#创建抑制神经元
def create_inhibitory_list(number,pro):
    p_list = [1] * (number - int(number * pro))
    n_list = [-1] * int(number * pro)
    p_list.extend(n_list)
    shuffle(p_list)
    return p_list


class Neurons:
    def __init__(self, dendrites, axon, inhib): #树突个数 轴突个数 以及抑制比
        self.dendrites = create_spine(dendrites,axon=False)
        self._inhib = create_inhibitory_list(len(self.dendrites), inhib)
        self.axon = create_spine(axon,axon=True)
        self._thred_m = int(len(self.dendrites)*0.5*round(1-inhib))
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
            #self.axon[i][-1] #需要转移的值  * self._inhib[i]
            self.axon[i][0] = self.axon[-1][-1]  # 设置值
        for i in range(len(self.axon)):
            self.axon[i].insert(0, self.axon[i].pop())  # 向右移动一格
        if self._voltage >= self._thred:
            self._thred = self._thred ** 2
            self.axon[-1][0] = 1
        else:
            self.axon[-1][0] = 0
            self._thred = self._thred - self._thred ** 0.5
            if self._thred <= self._thred_m:
                self._thred = self._thred_m


class Networks:
    def __init__(self, number, input_num, output_num, dendrites, axon, inhib):
        self.number = number
        self._dend_n = 0
        self._axon_n = 0
        self.input_list = []
        self.output_list= []
        self.self_link_id = []
        self.cell = []
        for i in range(self.number):
            _cell = Neurons(dendrites, axon, inhib)
            self.cell.append(_cell)
            self._dend_n = self._dend_n + len(_cell.dendrites)
            self._axon_n = self._axon_n + len(_cell.axon) - 1

        _input_list = list(range(self._dend_n))
        _output_list = list(range(self._axon_n))
        shuffle(_input_list)
        shuffle(_output_list)
        self.input_list = _input_list[0:input_num]
        self.output_list = _output_list[0:output_num]
        self.self_list = _output_list[
                            output_num:output_num + min(self._dend_n - input_num, self._axon_n - output_num)]
        self._axon_d = [0] * self._axon_n

    def tick(self):
        # print(self.input_list)
        # print(self.output_list)
        # print(self.self_link_id)
        self.self_transport()
        for i in range(self.number):
            self.cell[i].tick()

    def self_transport(self):
        count = 0
        for i in range(self.number):
            for j in range(len(self.cell[i].axon) - 1):
                self._axon_d[count] = self.cell[i].axon[j][-1]
                count = count + 1
        count = 0
        for i in range(self.number):
            for j in range(len(self.cell[i].dendrites)):
                if count in self.self_list:
                    self.cell[i].dendrites[j][0] = self._axon_d[self.self_list.index(count)] * self.cell[i]._inhib[j]
                count = count + 1

    def output(self):
        d = [0] * len(self.output_list)
        count = 0
        for i in range(self.number):
            for j in range(len(self.cell[i].axon) - 1):
                if count in self.output_list:
                   d[self.output_list.index(count)] = self.cell[i].axon[j][-1]
                   # print(count)
                count = count + 1
        return d

    def input(self, d):
        count = 0
        for i in range(self.number):
            for j in range(len(self.cell[i].dendrites)):
                if count in self.input_list:
                   self.cell[i].dendrites[j][0] = d[self.input_list.index(count)]
                   # print(count)
                count = count + 1
        return d


if __name__ == '__main__':
    seed(0)
    n = Networks(1, 5, 1, 5, 10, 0.1)

    a=[1,1,1,1,1]
    x= range(100)
    y=[]
    for i in x:
        n.input(a)
        n.tick()
        y.append(n.output()[0])
    print(n.__dict__)
    plt.plot(x, y)
    # plt.plot(x, z)
    plt.show()