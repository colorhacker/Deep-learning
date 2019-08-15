import neurons as N
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import user_mnist as udata
from random import randint, shuffle, seed, choice


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
            e = net_model.batch_code(e, 10)
            result[i] = pool.apply_async(func=net_model.batch_tick, args=(e,))
        pool.close()
        pool.join()
        for index, res in enumerate(result):
            plt.title(index)
            plt.bar(range(len(res.get())), res.get().flatten())
            # plt.plot(res.get())
            plt.show()
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    seed(0)
    np.random.seed(0)
    # 神经元个数，输入个数，树突最小长度，树突个数，突触长度，突触抑制率
    model = N.Networks(80, 784, 5, 20, 10, 0.1)
    model.info()
    # model.self_test(500, True)
    # model.update_threshold(0.1)
    # mnist_train = np.load("./temp/mnist_train.npy", allow_pickle=True)
    mnist_test = np.load("./temp/mnist_test.npy", allow_pickle=True)
    parallel_process(model, mnist_test)
    # parallel_process(model, mnist_train)
    # serial_process(model, mnist_train[0:1])
    # udata.mnist_class_save()