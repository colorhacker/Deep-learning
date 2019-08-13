import neurons as N
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
from random import randint, shuffle, seed, choice
from tqdm import tqdm
import os


if __name__ == '__main__':
    seed(0)
    # 神经元个数，输入个数，树突最小长度，树突个数，突触长度，突触抑制率
    model = N.Networks(50, 784, 5, 20, 10, 0.2)

    data = np.load("./temp/mnist_train.npy")
    for c in range(2):
        for i in tqdm(range(500)):
            for f in range(30):
                model.tick(data[c][i])
                # model.tick(data[0][i])
        print(model.active_freq())
        plt.plot(model.active_freq())
    plt.show()
