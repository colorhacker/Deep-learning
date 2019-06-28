from mnist import MNIST
from sklearn.cluster import KMeans,MiniBatchKMeans
from scipy.spatial import distance
import numpy as np
import os, random, base64, cv2

def load_mnist():
    data = MNIST('./python-mnist/data')
    image, label = data.load_training()
    return image, label

def display_image(data):
    #display_image(np.uint8(images[0], dtype=int).reshape(28, 28))
    cv2.imshow('img', data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#采用cosine返回最小距离的特征
def re_feature_matrix(c_means,data):
    m_len=0
    m_class=0
    for i in range(c_means.shape[0]):
        n_len = 0.5+0.5*np.corrcoef(c_means[i],data)[0][1]
        if n_len > m_len:
            m_len = n_len
            m_class = i
    return c_means[m_class]

#拆分一个矩阵分解为特征矩阵
def split_feature_data_c(data,c_w,c_h,s):
    result = np.empty(shape=[0,c_w*c_h], dtype=int)
    for x in range(0,data.shape[0]-c_w+1,s):
        for y in range(0,data.shape[1]-c_h+1,s):
            result = np.vstack((result, data[x:x+c_w,y:y+c_h].flatten()))
    return result

#重构数据
def rebuild_matrix_data(k_means,i_data,c_w,c_h,s):
    data = i_data.copy()
    for x in range(0,data.shape[0]-c_w+1,s):
        for y in range(0,data.shape[1]-c_h+1,s):
            #display_image(data[x:x + c_w, y:y + c_h])
            #display_image(re_feature_matrix(k_means,data[x:x + c_w, y:y + c_h].flatten()).reshape(c_w,c_h))
            data[x:x + c_w, y:y + c_h] = re_feature_matrix(k_means,data[x:x + c_w, y:y + c_h].flatten()).reshape(c_w,c_h)
    return data

images, labels =load_mnist()#加载数据

if os.path.exists("feature_file_L2_512.npy") == False:
    print("not found feature file .npy")
else:
    feature = np.load("feature_file_L2_512.npy").astype('uint8')

for i in range(1000):
    display_image(rebuild_matrix_data(feature,np.uint8(images[i], dtype=int).reshape(28, 28),7,7,7))
    #display_image(feature[i].reshape(7, 7))
