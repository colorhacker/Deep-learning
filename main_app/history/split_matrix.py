from mnist import MNIST
import datetime as dt
import numpy as np
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sort_func(x,y):
    return np.linalg.norm(x - y) #欧式距离
    # return 1 - spatial.distance.cosine(x, y) #cosine距离
    # a = 1-stats.pearsonr(x,y)[0] # Pearson product-moment correlation coefficients


#指定方式返回特征矩阵最接近的矩阵
def re_feature_same(feature, data):
    return np.array(sorted(feature, key=lambda element: sort_func(data, np.array(element))))[0]


#通过特征组合一个
def rebuild_matrix_for(feature, inpu_data, move_step, split_size):
    result = []
    for index in range(0, len(inpu_data), move_step):
        if (index + split_size) > len(inpu_data):
            break
        result.extend(re_feature_same(feature,inpu_data[index:index+split_size]))
    return np.array(result)


#拆分一个矩阵分解为特征
def partition_matrix_for(inpu_data, move_step, split_size):
    result = []
    for index in range(0, len(inpu_data), move_step):
        if (index + split_size) > len(inpu_data):
            break
        result.append(inpu_data[index:index+split_size])
    return np.array(result)


#删除相同的行
def delete_same_rows(data):
    new_array = [tuple(row) for row in data]
    uniques = np.unique(new_array, axis=0)
    print('delete ', data.shape[0] - uniques.shape[0],"same matrix")
    return uniques


#删除无效值的行
def delete_nan_rows(data):
    return np.delete(data, np.where(np.isnan(data))[0], axis=0)


#拆分一堆矩阵分解为特征
def partition_matrix(matirx, move_step, split_size):
    mkdir("./temp/")
    for i in range(0, int(matirx.shape[0]/1000), 1):
        result = partition_matrix_for(matirx[i*1000], move_step, split_size)
        print(dt.datetime.now(), i)
        for j in range(1, 1000, 1):
            result = np.vstack((result, partition_matrix_for(matirx[i*1000+j], move_step, split_size)))
        np.save("./temp/" + str(i), result)
    result = np.load("./temp/" + str(0)+".npy")
    for i in range(1, int(matirx.shape[0]/1000), 1):
        print("load ",i)
        result = np.vstack((result,np.load("./temp/" + str(i)+".npy")))
    np.save("./temp/feature_"+str(move_step)+"x"+str(split_size),delete_same_rows(result.astype('uint8')))
    for i in range(int(matirx.shape[0]/1000)):
        print("delete ", i)
        os.remove("./temp/" + str(i)+".npy")

if __name__=='__main__':
    images, labels = MNIST('./python-mnist/data', mode='vanilla', return_type='numpy').load_training()
    partition_matrix(images, 28, 56)