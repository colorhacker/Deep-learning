from mnist import MNIST
from sklearn.cluster import KMeans,MiniBatchKMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import os, random, base64, cv2

def load_mnist_training():
    data = MNIST('./python-mnist/data')
    image, label = data.load_training()
    return image, label

def load_mnist_testing():
    data = MNIST('./python-mnist/data')
    image, label = data.load_testing()
    return image, label

def display_image( data):
    cv2.imshow("image", cv2.resize(data, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#指定方式返回特征矩阵最接近的矩阵
def re_feature_matrix(center_data,data):
    m_class = 0
    m_len = np.linalg.norm(data-center_data[0])#L2欧式距离
    for i in range(1,center_data.shape[0],1):
        n_len = np.linalg.norm(data-center_data[i])#L2欧式距离
        if n_len < m_len:
            m_len = n_len
            m_class = i
    return m_class

#重构数据
def re_rebuild_data(k_means,i_data,c_w,c_h,s):
    re_d=[]
    for x in range(0,i_data.shape[0]-c_w+1,s):
        for y in range(0,i_data.shape[1]-c_h+1,s):
            re_d.append(re_feature_matrix(k_means,i_data[x:x + c_w, y:y + c_h].flatten()))
    return re_d

#重构数据
def rebuild_matrix_data(k_means,i_data,c_w,c_h,s):
    data = np.zeros((28,28))
    index=0
    for x in range(0,data.shape[0]-c_w+1,s):
        for y in range(0,data.shape[1]-c_h+1,s):
            data[x:x + c_w, y:y + c_h] = k_means[i_data[index]].reshape(c_w,c_h)
            index = index+1
    return data

mnist_feature = np.load("./mnist_feature.npy")
images, label = load_mnist_training()
image = np.array(images,dtype='float32')
result=[]
# for i in range(10):
for i in range(image.shape[0]):
    print(i)
    result.append(re_rebuild_data(mnist_feature,image[i].reshape(28, 28),7,7,7))
np.save("new_mnist_training",result)
