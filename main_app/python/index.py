import neurons as N
from multiprocessing import Pool
import matplotlib.pyplot as plt
import cv2 as opencv
import numpy as np
from mnist import MNIST
from random import randint, shuffle, seed
from tqdm import tqdm


def save_d():
    images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='lists').load_training()
    # images_t, labels_t = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_testing()
    digital = [[], [], [], [], [], [], [], [], [], []]
    for c in tqdm(range(len(images))):
        digital[labels[c]].append(images[c])
    for e in range(len(digital)):
        np.save("./temp/d/"+str(e), digital[e])

def encode_data(d):
    result = []
    for e in d:
        if e == 0:
            result.append([1, 1, 0, 1])
        else:
            result.append([0, 0, 1, 0])
    result = np.array(result).T
    return result.tolist()


def training(network, file, count):
    network.clear_evaluate()
    image = np.load(file)
    # for index in tqdm(range(len(image))):
    for index in tqdm(range(10)):
        e = encode_data(image[index])
        for c in range(count):
            for k in range(len(e)):
                network.input(e[k])
                network.tick()
    np.save("./temp/d/eva_"+str(), network.eva_active)
    return network.eva_active


if __name__ == '__main__':
    seed(0)
    net = N.Networks(50, 784, 100, 20, 20, 0.2)
    print("axon:", net.axon_num)
    print("dendrites:", net.dendrites_num)
    try:
        pool = Pool(10)
        for i in range(10):
            pool.apply_async(func=training, args=(N, "./temp/d/"+str(i)+".npy", 100))
        pool.close()
        pool.join()
    except:
        print("Error: unable to start process")