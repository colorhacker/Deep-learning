from mnist import MNIST
from sklearn.cluster import KMeans
import datetime as dt
import numpy as np
import os

def load_mnist():
    data = MNIST('./python-mnist/data')
    images, labels = data.load_training()
    return images, labels

#拆分一个矩阵分解为特征矩阵
def split_feature_data_c(data,c_w,c_h,s):
    result = np.empty(shape=[0,c_w*c_h], dtype=int)
    for x in range(0,data.shape[0]-c_w+1,s):
        for y in range(0,data.shape[1]-c_h+1,s):
            result = np.vstack((result, data[x:x+c_w,y:y+c_h].flatten()))
    return result

save_path = "./split/f_"
#拆分一堆矩阵分解为特征
def split_feature_data(c_w,c_h,s):
    images, labels = load_mnist()
    image = np.array(images, dtype='float32')
    for i in range(0, 60, 1):
        result = np.empty(shape=[0, c_w * c_h])
        for j in range(0, 1000, 1):
            print(dt.datetime.now(), j)
            result = np.vstack((result, split_feature_data_c(image[i*1000+j].reshape(28, 28), c_w, c_h, s)))
        np.save(save_path+ str(i), result)
    result = np.load(save_path + str(0)+".npy")
    for i in range(1, 60, 1):
        print("load ",i)
        result = np.vstack((result,np.load(save_path + str(i)+".npy")))
    np.save(save_path+str(c_w)+"x"+str(c_h)+"x"+str(s),result.astype('uint8'))
    for i in range(60):
        print("delete ", i)
        os.remove(save_path + str(i)+".npy")

def delete_same_rows(data):
    print('input data:',data.shape)
    new_array = [tuple(row) for row in data]
    uniques = np.unique(new_array, axis=0)
    print('output data:', uniques.shape)
    return uniques

# np.save("./split_feature/split_feature_7x7x7_t",delete_same_rows(np.load("./split_feature/split_feature_7x7x7.npy")))
# np.save("./split_feature/split_feature_7x7x5_t",delete_same_rows(np.load("./split_feature/split_feature_7x7x5.npy")))
# np.save("./split_feature/split_feature_7x7x4_t",delete_same_rows(np.load("./split_feature/split_feature_7x7x4.npy")))
# np.save("./split_feature/split_feature_7x7x3_t",delete_same_rows(np.load("./split_feature/split_feature_7x7x3.npy")))
# np.save("./split_feature/split_feature_7x7x2_t",delete_same_rows(np.load("./split_feature/split_feature_7x7x2.npy")))

# split_feature_data(14,14,2) #分解图像