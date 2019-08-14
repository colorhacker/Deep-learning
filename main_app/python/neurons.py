
from tqdm import tqdm
from random import randint, shuffle, seed ,sample
import matplotlib.pyplot as plt
import numpy as np

cell_threshold_max = 0.4  # 设置阈值和最大输入值比例
cell_threshold_attenuate = 0.15  # 衰减系数 0.05需要衰减10次才能到最大值
cell_blank_synapse = 0.0  # 树突无连接的数量比


# cell_num 神经元个数
# input_number 输入的数据个数
# sheath_length 单个轴突有多少个分支
# cell_dendrite 单个神经元内部树突个数
# axon_length 轴突长度
# inhibit 突触抑制比例
class Networks:
    def __init__(self, cell_number, input_number, branch_length, cell_dendrite, axon_length, inhibit):
        self.cell_num = cell_number  # 神经元个数
        self.input_num = input_number  # 输入数据个数
        self.dendrites_table = [randint(cell_dendrite, int(cell_dendrite * 1.5)) for _ in range(self.cell_num)]  # 单个神经元拥有的树突表
        self.dendrites_number = int(sum(self.dendrites_table) * (1 + cell_blank_synapse))  # 计算总的树突个数
        self.input_table = sample(list(range(self.dendrites_number)), input_number)  # 生成input 对应表
        self.cell_threshold_fix = [_*cell_threshold_max for _ in self.dendrites_table]  # 每个神经元的阈值
        self.cell_threshold = self.cell_threshold_fix  # 当前神经元的阈值
        self.synapse_offset = [randint(branch_length, int(branch_length * 1.5)) for _ in range(self.dendrites_number)]  # 树突长度值 也是数组的偏移量
        self.synapse = np.zeros(shape=(self.dendrites_number, max(self.synapse_offset) + 1))  # 树突
        self.axon = [[0] * axon_length for _ in range(cell_number)]  # 突触
        self.axon_dendrites_list = list(range(self.dendrites_number))  # 突触与与髓鞘的连接表
        shuffle(self.axon_dendrites_list)
        self.synapse_polarity = [1] * self.dendrites_number  # 树突的极性
        self.synapse_polarity[0:int(self.dendrites_number*inhibit)] = [-1]*int(self.dendrites_number*inhibit)
        shuffle(self.synapse_polarity)
        self.synapse_polarity = np.array(self.synapse_polarity)
        np.put(self.synapse_polarity, self.input_table, [1]*self.input_num)
        self.synapse_polarity = self.synapse_polarity.reshape(self.dendrites_number, 1)
        self.notes_tick_active = np.zeros(shape=(self.cell_num, 1))  # 用于保存神经元的激活值
        self.notes_tick_count = 1

    def update_input(self, data):
        self.synapse = np.roll(self.synapse, -1)  # 移动髓鞘数据
        offset_index = 0
        for index in range(self.cell_num):
            for count in range(self.dendrites_table[index]):  # 按照树突进行值的设置
                self.synapse[offset_index, self.synapse_offset[offset_index]] = self.axon[index][0]
                offset_index += 1

        for index, count in enumerate(self.input_table):  # 再进行input值设置
            self.synapse[index, self.synapse_offset[index]] = data[index]

    def update_output(self):
        self.axon = np.roll(self.axon, -1)  # 移动突触数据
        count_line = 0
        for index, count in enumerate(self.dendrites_table):  # 通过树突更新髓鞘起始数据
            if sum(self.synapse[count_line:count_line + count, 0:1] *
                   self.synapse_polarity[count_line:count_line + count, :]) \
                    > self.cell_threshold[index]:
                self.cell_threshold[index] += count
                self.axon[index][-1] = 1
            else:
                self.axon[index][-1] = 0
                self.cell_threshold[index] -= self.cell_threshold[index] * cell_threshold_attenuate
                if self.cell_threshold[index] <= self.cell_threshold_fix[index]:
                    self.cell_threshold[index] = self.cell_threshold_fix[index]
            count_line += count

    def tick(self, d):
        self.notes_tick_count += 1
        self.update_input(d)
        self.update_output()
        self.notes_tick_active += self.axon[:, 0:1]  # 记录神经元的激活次数

    def batch_tick(self, d):
        for e in tqdm(d):
            self.tick(e)
        return self.active_freq()

    def active_freq(self):
        return self.notes_tick_active / self.notes_tick_count  # 计算激活频率

    def clean_freq(self):
        self.notes_tick_count = 1
        self.notes_tick_active = np.zeros(shape=(self.cell_num, 1))

    def update_threshold(self, biase):
        index = self.active_freq().tolist().index(min(self.active_freq()))
        self.cell_threshold_fix[index] -= biase
        

if __name__ == '__main__':
    seed(0)
    net = Networks(2, 2, 4, 3, 4, 0.2)
    for i in range(40):
        net.tick([2, 2])
    print(net.active_freq())
    print(net.update_threshold(0.1))
