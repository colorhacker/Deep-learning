from mnist import MNIST
from multiprocessing import Pool
from scipy import spatial,stats
from tqdm import tqdm
import split_matrix as splitm
import numpy as np
import os


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