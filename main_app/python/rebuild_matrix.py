from mnist import MNIST
from multiprocessing import Pool
from scipy import spatial,stats
from sklearn.cluster import KMeans
from tqdm import tqdm
import split_matrix as splitm
import numpy as np
import os


def kmeans_process(n_class,data):
    kmeans = KMeans(n_clusters=n_class,init='random',algorithm='full').fit(data)
    return kmeans.cluster_centers_


#自定义排序矩阵
def custum_sort_matrix(data):
    data_list = data.tolist()
    return np.array(sorted(data_list, key=lambda x:np.linalg.norm(np.zeros(data.shape[1]) - np.array(x))))


#拼接矩阵显示
def parallel_matrix(data,count,space=1):
    step  = int(data.shape[1]**0.5)
    image = np.zeros(shape=[count*(step+space), count*(step+space)])
    index = 0
    for x in range(0,image.shape[0],step+space):
        for y in range(0,image.shape[1],step+space):
            image[x:x + step, y:y + step] = data[index].reshape(step,step)
            index = index + 1
    return image

def rebuilt_feature(thread,feature, matrix, move_step, split_size, path, start, stop):
    result = []
    if thread == 0:
        for i in tqdm(range(start, stop, 1)):
            result.append(splitm.rebuild_matrix_for(feature, matrix[i], move_step, split_size))
    else:
        for i in range(start, stop, 1):
            result.append(splitm.rebuild_matrix_for(feature, matrix[i], move_step, split_size))
    np.save(path+str(start)+"_"+str(stop), result)


def rebuild_feature_matrix(feature,data,async_count,save_path):
    step = int(data.shape[0]/async_count)
    try:
        print('Parent process %s.' % os.getpid())
        pool = Pool(async_count)
        for i in range(0,data.shape[0],step):
            pool.apply_async(func=rebuilt_feature,args=(i, feature, data, 56, 56, "./temp/_", i, i+step))
        pool.close()
        pool.join()
        train_data = []
        for i in range(0, data.shape[0], step):
            train_data.extend(np.load("./temp/_" + str(i) + "_" + str(i + step) + ".npy"))
            os.remove("./temp/_" + str(i) + "_" + str(i + step) + ".npy")
        np.save(save_path,np.array(train_data))
    except:
        print("Error: unable to start process")


def rebuild_mnist(file):
    feature_matrix = np.load(file)
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_training()
    rebuild_feature_matrix(feature_matrix,images, 20, "./temp/train_feature")
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_testing()
    rebuild_feature_matrix(feature_matrix,images, 20, "./temp/test_feature")


if __name__ == '__main__':
    rebuild_mnist("./temp/kmeans_feature_28x56_512.npy")