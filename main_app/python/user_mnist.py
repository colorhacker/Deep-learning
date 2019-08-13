import neurons as N
import matplotlib.pyplot as plt
# import sort_data as sortd
from multiprocessing import Pool, Queue, Process, current_process
# import cv2 as opencv
import numpy as np
from mnist import MNIST
from random import randint, shuffle, seed, choice
from tqdm import tqdm
import os


def sort_func(x,y):
    return np.linalg.norm(x - y) # 欧式距离
    # return 1 - spatial.distance.cosine(x, y) #cosine距离
    # a = 1-stats.pearsonr(x,y)[0] # Pearson product-moment correlation coefficients


# 指定方式返回特征矩阵最接近的矩阵
def re_feature_same(feature, data):
    return np.array(sorted(range(len(feature)), key=lambda element: sort_func(data, np.array(feature[element]))))[0]


# 自定义排序矩阵1d
def custum_sort_list(data, rule=False):
    target_data = 0
    data_list = data.tolist()
    if rule:
        # target_data = np.array(sorted(data_list, key=lambda element: sort_func(target_data , np.array(element))))
        target_data = np.array(sorted(data_list, key=lambda element: sort_func(target_data , np.array(element))))[len(data_list)-1]
    value = np.array(sorted(data_list, key=lambda element: sort_func(target_data , np.array(element))))
    labels = np.array(sorted(range(len(data_list)), key=lambda element: sort_func(target_data , np.array(data_list[element]))))
    return value,labels


# 按类别存储手写数据
def mnist_class_save():
    images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='lists').load_training()
    digital = [[], [], [], [], [], [], [], [], [], []]
    for c in tqdm(range(len(images))):
        digital[labels[c]].append(np.array(images[c]).astype("uint8"))
    np.save("./temp/mnist_train", digital)

    images_t, labels_t = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_testing()
    digital = [[], [], [], [], [], [], [], [], [], []]
    for c in tqdm(range(len(images_t))):
        digital[labels_t[c]].append(np.array(images_t[c]).astype("uint8"))
    np.save("./temp/mnist_test", digital)

def sort_bar_show(title, data):
    value, labels = custum_sort_list(data, rule=True)
    plt.bar(range(len(value)), value)
    # plt.plot(value)
    plt.xticks(range(len(value)), labels, rotation='vertical')
    plt.title(title)
    plt.show()


class Realpolt:
    def __init__(self, title, delay):
        # plt.xticks(range(len(value)), labels, rotation='vertical')
        self.title = title
        self.delay = delay
        self.Queue = Queue()

        plt.title(self.title)
        self.Process = Process(target=self._p, args=())
        self.Process.start()

    def put(self, d):
        self.Queue.put(d)

    def _p(self):
        while True:
            data = self.Queue.get()
            plt.plot(data, color='red')
            plt.pause(self.delay)