from mnist import MNIST
from multiprocessing import Pool
from scipy import spatial,stats
import sort_data as sortd
import numpy as np
import os


dist_index = 0 #设置迭代相似度矩阵

#指定方式返回特征矩阵最接近的矩阵
def re_feature_matrix(center_data,data):
    global dist_index
    return np.array(sorted(range(len(center_data)), key=lambda element: sortd.sort_func(data, np.array(center_data[element]))))[dist_index]


#重构数据
def rebuild_for(k_means,i_data,c_w,c_h,step):
    re_d=[]
    for x in range(0,i_data.shape[0]-c_w+1,step):
        for y in range(0,i_data.shape[1]-c_h+1,step):
            re_d.append(re_feature_matrix(k_means,i_data[x:x + c_w, y:y + c_h].flatten()))
    return re_d


def rebuilt_feature(thread,feature,matrix,p_width,p_height,step,path,start,stop):
    width_height = int(matrix.shape[1]**0.5)
    if width_height**2 != matrix.shape[1]:
        raise RuntimeError('matirx width not equal height')
    if matrix.shape[0] < 1000:
        raise RuntimeError('matirx number need more than 1000')
    result=[]
    for i in range(start,stop,1):
        print(thread,i)
        result.append(rebuild_for(feature,matrix[i].reshape(width_height, width_height),p_width,p_height,step))
    np.save(path+str(start)+"_"+str(stop),result)


def rebuild_feature_matrix(data,feature,async_count,save_path):
    step = int(data.shape[0]/async_count)
    try:
        print('Parent process %s.' % os.getpid())
        pool = Pool(async_count)
        for i in range(0,data.shape[0],step):
            pool.apply_async(func=rebuilt_feature,args=(str(i), feature, data, 7, 7, 7, "./temp/_", i, i+step))
        pool.close()
        pool.join()
        train_data = []
        for i in range(0, data.shape[0], step):
            if i is 0:
                train_data = np.load("./temp/_" + str(i) + "_" + str(i + step) + ".npy")
            else:
                train_data = np.vstack((train_data, np.load("./temp/_" + str(i) + "_" + str(i + step) + ".npy")))
            os.remove("./temp/_" + str(i) + "_" + str(i + step) + ".npy")
        np.save(save_path,train_data)
    except:
        print("Error: unable to start process")


#重构数据
def rebuild_matrix_c(feature,f_array,c_width,c_hight,step):
    wh = int(len(f_array) ** 0.5)
    data = np.zeros((wh*c_width,wh*c_hight))
    index=0
    for x in range(0,data.shape[0],step):
        for y in range(0,data.shape[1],step):
            data[x:x + c_width, y:y + c_hight] = feature[f_array[index]].reshape(c_width,c_hight)
            index = index+1
    return data


def rebuild_mnist(file):
    global dist_index
    feature_matrix = np.load(file)
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_training()
    rebuild_feature_matrix(images,feature_matrix,20,"./temp/train_feature"+str(dist_index)+"_")
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_testing()
    rebuild_feature_matrix(images,feature_matrix,20,"./temp/test_feature"+str(dist_index)+"_")

if __name__ == '__main__':
    for i in range(5):
        dist_index = i
        rebuild_mnist("./temp/kmeans_feature_7x7x7_512.npy")