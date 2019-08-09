import neurons as N
import matplotlib.pyplot as plt
# import sort_data as sortd
from multiprocessing import Pool
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


def fit_c(image, network, count):
    network.clear_evaluate()
    e = N.encode_data(image)
    for c in range(count):
        for k in e:
            network.input(k)
            network.tick()
    return network.eva_active


def training(s_id, network, start, end):
    network.clear_evaluate()
    image = np.load("./temp/mnist_train.npy", allow_pickle=True)[s_id]
    # for index in tqdm(range(len(image))):
    for index in tqdm(range(start, end)):
        e = N.encode_data(image[index])
        for c in range(50):
            for k in range(len(e)):
                network.input(e[k])
                network.tick()
    np.save("./temp/train_"+str(s_id), network.eva_active)
    return network.eva_active


def training_random(network, update):
    images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='lists').load_training()
    network.clear_evaluate()
    update_count = update
    means = 0
    for index in tqdm(range(len(images))):
        e = N.encode_data(images[index])
        for c in range(10):
            for k in range(len(e)):
                network.input(e[k])
                network.tick()
        update_count = update_count - 1
        if update_count == 0:
            update_count = update
            means = network.update_thred_m(1)
            print(means)
            network.clear_evaluate()
        if means > 0.8:
            break


def testing(s_id, network,  start, end):
    network.clear_evaluate()
    feature = np.load("./temp/train_total.npy")
    feature = feature / feature.max()
    image = np.load("./temp/mnist_test.npy", allow_pickle=True)[s_id]
    success = 0
    # for index in range(start, end):
    for index in tqdm(range(start, end)):
        result = fit_c(image[index], network, 30)
        result = np.array(result)
        result = result / result.max()
        _class = re_feature_same(feature, result)
        if _class == s_id:
            success = success + 1
    return (success/(end - start)) * 100


def for_training(network):
    for i in range(10):
        training(i, network, 0, 100)
    total = []
    for e in range(10):
        total.append(np.load("./temp/train_"+str(e)+".npy"))
        os.remove("./temp/train_"+str(e)+".npy")
    np.save("./temp/train_total", total)
    np.savetxt("./temp/train_total.csv", (total/total.max()).T, delimiter=',')


def pool_training(network):
    try:
        pool = Pool(10)
        for i in range(10):
            pool.apply_async(func=training, args=(i, network, 0, 100))
        pool.close()
        pool.join()
        total = []
        for e in range(10):
            total.append(np.load("./temp/train_" + str(e) + ".npy"))
            os.remove("./temp/train_" + str(e) + ".npy")
        np.save("./temp/train_total", total)
        np.savetxt("./temp/train_total.csv", (total/total.max()).T, delimiter=',')
    except:
        print("Error: unable to start process")


def bar_show(title):
    if os.path.exists("./temp/train_total.npy"):
        d = np.load("./temp/train_total.npy")
        np.savetxt("./temp/train_total.csv", (d / d.max()).T, delimiter=',')
        d = d/d.max()
        for e in range(10):
            sort_bar_show(title+str(e), d[e])


def sort_bar_show(title, data):
    value, labels = custum_sort_list(data, rule=True)
    plt.bar(range(len(value)), value)
    # plt.plot(value)
    plt.xticks(range(len(value)), labels, rotation='vertical')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    seed(0)
    net = N.Networks(40, 784, 0, 25, 15, 0.3)
    net.self_update_thred_m(5)  # 更新阈值
    print("dendrites:", net.dendrites_num)
    print("axon:", net.axon_num)
    print("selt:", net.dendrites_num - 784)

    bar_show("d")
    # pool_training(net)
    # for_training(net)
    # bar_show("train ")
    # training_random(net, 0, 100)
    # training_random(net, 100, 200)
    # result = []
    # for i in range(10):
    #     result.append(testing(i, net, 0, 10))
    # print(np.array(result).sum()/10)
    # testing_c(0, 0, net, 100)


