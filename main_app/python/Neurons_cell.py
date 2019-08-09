
from tqdm import tqdm
from random import randint, shuffle, seed
import matplotlib.pyplot as plt
import numpy as np

axon_myelin_sheath = 6  # 髓鞘 长度
dendrite_length = 3  # 树突长度
cell_threshold_max = 0.45  # 设置阈值和最大输入值比例
cell_threshold_attenuate = 0.15  # 衰减系数 0.05需要衰减10次才能到最大值


# cell_num 神经元个数
# input_number 输入的数据个数
# sheath_length 单个轴突有多少个分支
# dendrite_c 单个神经元内部树突个数
# axon_length 轴突长度
# inhibit 突触抑制比例
class Networks:
    def __init__(self, cell_number, input_number, sheath_length, dendrite_c, axon_length, inhibit):
        self.cell_num = cell_number
        self.input_num = input_number
        self.dendrites_table = [randint(dendrite_c, int(dendrite_c * 1.5)) for _ in range(self.cell_num)]
        self.dendrites_table.append(input_number)
        self.dendrites_number = sum(self.dendrites_table)
        self.synapse = [[0] * randint(sheath_length, int(sheath_length * 1.5)) for _ in range(self.dendrites_number)]
        self.axon = [[0] * axon_length for _ in range(cell_number)]
        self.axon_dendrites_list = list(range(self.dendrites_number))
        shuffle(self.axon_dendrites_list)
        self.synapse_polarity = [1] * self.dendrites_number
        self.synapse_polarity[0:int(self.dendrites_number*inhibit)] = [-1]*int(self.dendrites_number*inhibit)
        shuffle(self.synapse_polarity)

        # def tick(self):
        #
        # def self_transport(self):
        #
        # def output(self):
        #
        # def input(self):
        #
        # def self_evaluate(self):
        #
        # def clear_evaluate(self):
        #
        # def self_update_thred_m(self):
        #
        # def update_thred_m(self):


if __name__ == '__main__':
    net = Networks(10, 10, 4, 3, 10, 0.2)
    print(net.dendrites_number)
    print(net.synapse)
    print(net.axon_dendrites_list)
    print(net.synapse_polarity)
