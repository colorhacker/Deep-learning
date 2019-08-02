
from random import randint,shuffle


class Dendrites:
    def __init__(self, synapse, length):
        self.spine = []
        self.energy = 0
        self.count = randint(1,length)
        for i in range(self.count):
            self.spine.append([0]*randint(1,int(self.count**0.5)))

    def cell_tick(self):
        for i in range(len(self.spine)):
            self.spine[i].insert(0,self.spine[i].pop()) #向右移动一格
            self.spine[i][0] = 0 #清除最左边值
            self.energy = self.energy + self.spine[i][-1]

class Neurons:
    def __init__(self, dendrites, axon):
        self.dendrites = dendrites #神经元前级树突
        self.energy = 0  #神经元中心的能量值
        self.threshold = 0 #神经元中心的阈值
        self.axon = axon #神经元后级轴突

    def cell_tick(self):
        if sum(self.dendrites.energy) > self.threshold:
            self.energy = sum(self.dendrites)
        self.threshold = self.energy**0.5


class Axon:
    def __init__(self, length):
        self.spine = []
        self.count = randint(length,length+int(length**0.5))
        for i in range(self.count):
            self.spine.append([0]*(length+randint(1,int(self.count**0.5))))

    def cell_tick(self):
        for i in range(len(self.spine)):
            self.spine[i].insert(0,self.spine[i].pop()) #向右移动一格
            self.spine[i][0] = 0 #清除最左边值


class Synapse:
    def __init__(self, axon_number, dendrites_number, ipsp):
        self.count = max(axon_number, dendrites_number)
        self.ipsp = list(range(self.count))
        shuffle(self.ipsp)
        self.ipsp = self.ipsp[0:int(self.count*ipsp)]
        self.link_list = list(range(self.count))
        shuffle(self.link_list)

    # def cell_tick(self):


if __name__ == '__main__':
    a=Synapse(10,20,0.5)
    print(a.ipsp)