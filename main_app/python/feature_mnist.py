from mnist import MNIST
import datetime as dt
import numpy as np

def load_mnist():
    data = MNIST('./python-mnist/data')
    images, labels = data.load_training()
    return images, labels

def mnist_to_matrix(data):
    return np.uint8(data, dtype=int).reshape(28, 28)

#拆分一个矩阵分解为特征矩阵
def split_feature_data_c(data,c_w,c_h,s):
    result = np.empty(shape=[0,c_w*c_h], dtype=int)
    for x in range(0,data.shape[0]-c_w+1,s):
        for y in range(0,data.shape[1]-c_h+1,s):
            result = np.vstack((result, data[x:x+c_w,y:y+c_h].flatten()))
    return result

#拆分一堆矩阵分解为特征
def split_feature_data(c_w,c_h,s):
    images, labels = load_mnist()
    for i in range(0, 60, 1):
        result = np.empty(shape=[0, c_w * c_h], dtype=int)
        for j in range(0, 1000, 1):
            print(dt.datetime.now(), j)
            result = np.vstack((result, split_feature_data_c(mnist_to_matrix(images[i*1000+j]), c_w, c_h, s))).astype('uint8')
        np.save("./feature/feature",result)
    return 0

split_feature_data(7,7,1) #分解图像