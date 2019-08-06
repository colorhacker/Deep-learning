
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
    def __init__(self, dendrites, axon, inhib):
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


if __name__ == '__main__':
    seed(0)
    n1 = Neurons(10, 5, 0.1)
    print(n1.__dict__)
    x = range(100)
    y = []
    z = []
    for i in x:
        n1.input([randint(0, 1) for _ in range(100)])
        n1.tick()
        y.append(sum(n1.output()))
        z.append(n1.axon[-1][-1])

    # plt.plot(x, y)
    plt.plot(x, z)
    plt.show()