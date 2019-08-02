from tqdm import tqdm
import matplotlib.pyplot as plt
import rebuild_matrix as rebulid
import sort_data as sortd
import split_matrix as splitm

import cv2 as opencv
import numpy as np
from mnist import MNIST
from sklearn.cluster import KMeans

def display_image(data):
    opencv.imshow("image", opencv.resize(data, None, fx=2, fy=2, interpolation=opencv.INTER_CUBIC))
    opencv.waitKey(0)
    opencv.destroyAllWindows()


def save_image(matrix):
    opencv.imwrite("./temp/test",opencv.resize(matrix, None, fx=2, fy=2, interpolation=opencv.INTER_CUBIC))


def kmeans_process(n_class,data):
    kmeans = KMeans(n_clusters=n_class,init='random',algorithm='full').fit(data)
    return kmeans.cluster_centers_

#自定义排序矩阵
def custum_sort_matrix(data):
    data_list = data.tolist()
    return np.array(sorted(data_list, key=lambda x:np.linalg.norm(np.zeros(data.shape[1]) - np.array(x))))


def parallel_matrix(data,count,space=1):
    step  = int(data.shape[1]**0.5)
    image = np.zeros(shape=[count*(step+space), count*(step+space)])
    index = 0
    for x in range(0,image.shape[0],step+space):
        for y in range(0,image.shape[1],step+space):
            image[x:x + step, y:y + step] = data[index].reshape(step,step)
            index = index + 1
    return image


if __name__ == '__main__':
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_training()
    images_t, labels_t = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_testing()
    # image = parallel_matrix(feature,10)
    # plt.matshow(images[59999].reshape(28,28))
    # plt.show()
    # print(splitm.sort_func(images,np.load("./temp/train_feature_49_49.npy"))/60000)
    # print(splitm.sort_func(images,np.load("./temp/train_feature_28_56.npy"))/60000)
    # print(images.shape)
    # print(np.load("./temp/train_feature_28_56.npy").shape)
