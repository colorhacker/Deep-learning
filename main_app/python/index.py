import neurons as N
import matplotlib.pyplot as plt
import cv2 as opencv
import numpy as np
from mnist import MNIST
from random import randint, shuffle, seed


def Encode_vaule(d):
    result = []
    for e in d:
        if e == 0:
            result.append([1, 1, 1, 1])
        else:
            result.append([1, 1, 0, 1])
    result = np.array(result).T
    return result.tolist()


if __name__ == '__main__':
    seed(0)
    # images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='lists').load_training()
    # images_t, labels_t = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_testing()

    image_0 = np.load("./temp/image_0.npy").tolist()
    image_1 = np.load("./temp/image_1.npy").tolist()
    net = N.Networks(20, 784, 10, 50, 50, 0.2)
    for i in range(1):
        dd = Encode_vaule(image_0[i])
        # print(dd)
        x = range(500)
        y = []
        for i in x:
            print(i)
            for t in range(4):
                net.input(dd[t])
                net.tick()
            y.append(sum(net.output()))
        plt.plot(x, y)
        # plt.plot(x, z)
        plt.show()
