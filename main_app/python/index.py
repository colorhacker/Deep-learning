import neurons as N
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from multiprocessing import Pool, Queue, Process, current_process
from mnist import MNIST
from random import randint, shuffle, seed, choice
from tqdm import tqdm


def serial_process(net_model, train_data):
    for a in train_data:
        net_model.batch_tick(a)
        plt.plot(net_model.active_freq())
        net_model.clean_freq()
    plt.show()


def parallel_process(net_model, train_data):
    try:
        pool = Pool(len(train_data))
        result = list([[]]*len(train_data))
        for i, e in enumerate(train_data):
            result[i] = pool.apply_async(func=net_model.batch_tick, args=(e,))
        pool.close()
        pool.join()
        for res in result:
            plt.bar(range(len(res.get())),res.get().flatten())
            # plt.plot(res.get())
            plt.show()
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    seed(0)
    # 神经元个数，输入个数，树突最小长度，树突个数，突触长度，突触抑制率
    model = N.Networks(50, 784, 5, 10, 10, 0.2)
    model.info()
    model.self_test(1000, True)
    # model.update_threshold(0.1)
    # mnist_train = np.load("./temp/mnist_train.npy")
    mnist_test = np.load("./temp/mnist_test.npy")
    parallel_process(model, mnist_test)
    # parallel_process(model, mnist_train)
    # serial_process(model, mnist_train)



