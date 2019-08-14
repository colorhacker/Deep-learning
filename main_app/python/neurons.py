
from tqdm import tqdm
from random import randint, shuffle, seed, sample
from multiprocessing import Pool, Queue, Process, current_process
import matplotlib.pyplot as plt
import numpy as np

soma_threshold_pro = 0.45  # 设置阈值和最大输入值比例
soma_threshold_attenuate = 0.15  # 衰减系数 0.05需要衰减10次才能到最大值
soma_blank_synapse = 0.1  # 树突无连接的数量比


# cell_num 神经元个数
# input_number 输入的数据个数
# sheath_length 单个轴突有多少个分支
# cell_dendrite 单个神经元内部树突个数
# axon_length 轴突长度
# inhibit 突触抑制比例
class Networks:
    def __init__(self, soma_number, input_number, branch_length, cell_dendrite, axon_length, inhibit):
        self.soma_num = soma_number  # 神经元个数
        self.input_num = input_number  # 输入数据个数
        self.dendrites_table = [randint(cell_dendrite, int(cell_dendrite * 1.5)) for _ in range(self.soma_num)]  # 单个神经元拥有的树突表
        self.dendrites_number = int(sum(self.dendrites_table) * (1 + soma_blank_synapse))  # 计算总的树突个数
        self.input_table = sample(list(range(self.dendrites_number)), input_number)  # 生成input 对应表
        self.soma_threshold_fixed = [_*soma_threshold_pro for _ in self.dendrites_table]  # 每个神经元的阈值
        self.soma_threshold = self.soma_threshold_fixed.copy()  # 当前神经元的阈值
        self.synapse_offset = [randint(branch_length, int(branch_length * 1.5)) for _ in range(self.dendrites_number)]  # 树突长度值 也是数组的偏移量
        self.synapse = np.zeros(shape=(self.dendrites_number, max(self.synapse_offset) + 1))  # 树突
        self.axon = [[0] * axon_length for _ in range(self.soma_num)]  # 突触
        self.axon_dendrites_list = list(range(self.dendrites_number))  # 突触与与髓鞘的连接表
        shuffle(self.axon_dendrites_list)
        self.synapse_polarity = [1] * self.dendrites_number  # 树突的极性
        self.synapse_polarity[0:int(self.dendrites_number*inhibit)] = [-1]*int(self.dendrites_number*inhibit)
        shuffle(self.synapse_polarity)
        self.synapse_polarity = np.array(self.synapse_polarity)
        np.put(self.synapse_polarity, self.input_table, [1]*self.input_num)  # 输入的数据设置不可抑制
        self.synapse_polarity = self.synapse_polarity.reshape(self.dendrites_number, 1)
        self.notes_tick_active = np.zeros(shape=(self.soma_num, 1))  # 用于保存神经元的激活值
        self.notes_tick_count = 0

    def info(self):
        print("input number         :%d" % self.input_num)
        print("smon number          :%d" % self.soma_num)
        print("dendrites number     :%d" % self.dendrites_number)
        print("self join number     :%d" % (self.dendrites_number - self.input_num))
        print("axon length          :%d" % len(self.axon[0]))
        print("dendrite length      :%d - %d" % (min(self.synapse_offset), max(self.synapse_offset)))
        print("blank dendrites      :%d" % (int(sum(self.dendrites_table) * soma_blank_synapse)))
        print("smon threshold pro   :%0.2f" % soma_threshold_pro)
        print("smon attenuate pro   :%0.2f" % soma_threshold_attenuate)
        print("estimated frequency  :%0.2f" % (soma_threshold_attenuate/soma_threshold_pro))

    def update_input(self, data):
        self.synapse = np.roll(self.synapse, -1)  # 移动髓鞘数据
        self.synapse[:, -1] = 0  # 清除roll移动的移位数据
        offset_index = 0
        for index in range(self.soma_num):
            for count in range(self.dendrites_table[index]):  # 按照树突进行值的设置
                _p = self.axon_dendrites_list[offset_index]
                self.synapse[_p, self.synapse_offset[_p]] = self.axon[index][0]
                offset_index += 1
        for index, count in enumerate(self.input_table):  # 再进行input值设置
            _p = self.axon_dendrites_list[count]
            self.synapse[_p, self.synapse_offset[_p]] = data[index]

    def update_output(self):
        self.axon = np.roll(self.axon, -1)  # 移动突触数据
        count_line = 0
        for index, count in enumerate(self.dendrites_table):  # 通过树突更新髓鞘起始数据
            if sum(self.synapse[count_line:count_line + count, 0:1] *
                   self.synapse_polarity[count_line:count_line + count, :]) \
                    > self.soma_threshold[index]:
                self.soma_threshold[index] += count
                self.axon[index][-1] = 1
            else:
                self.axon[index][-1] = 0
                self.soma_threshold[index] -= self.soma_threshold[index] * soma_threshold_attenuate
                if self.soma_threshold[index] <= self.soma_threshold_fixed[index]:
                    self.soma_threshold[index] = self.soma_threshold_fixed[index]
            count_line += count

    def tick(self, d):  # 调用一次
        self.notes_tick_count += 1
        self.update_input(d)
        self.update_output()
        self.notes_tick_active += self.axon[:, 0:1]  # 记录神经元的激活次数

    def batch_tick(self, d):  # 批量调用
        for e in tqdm(d):
            self.tick(e)
        return self.active_freq()

    def active_freq(self):
        return self.notes_tick_active / self.notes_tick_count  # 计算激活频率

    def clean_freq(self):
        self.notes_tick_count = 1
        self.notes_tick_active = np.zeros(shape=(self.soma_num, 1))

    def self_test(self, number, graph=False):
        self.clean_freq()
        self.batch_tick(np.random.randint(0, 2, (number, self.input_num)))
        if graph:
            plt.title("random")
            # plt.plot(range(self.soma_num), self.active_freq().flatten())
            plt.bar(range(self.soma_num), self.active_freq().flatten())
            plt.show()
        return self.active_freq()

    def update_threshold(self, baise):
        index = self.active_freq().tolist().index(min(self.active_freq()))
        self.soma_threshold_fixed[index] -= baise


if __name__ == '__main__':
    seed(0)
    net = Networks(2, 2, 4, 3, 4, 0.5)
    print(net.__dict__)
    for i in range(40):
        net.tick([2, 2])
        print(net.__dict__)
    print(net.active_freq())
    print(net.update_threshold(0.1))
