from mnist import MNIST
from sklearn.cluster import KMeans,MiniBatchKMeans
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

def flatten_to_matrix(data):
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
    image, label = load_mnist()
    result = np.empty(shape=[0,c_w*c_h], dtype=int)
    #for i in range(0,len(images),1):
    for i in range(0, 60, 1):
        result = np.vstack((result, split_feature_data_c(flatten_to_matrix(image[i]),c_w,c_h,s)))
    return result

#重构数据
def rebuild_matrix_data(kmeans,data,c_w,c_h,s):
    fit_array = kmeans.predict(split_feature_data_c(data, c_w,c_h,s))
    fit_index = 0
    for x in range(0,data.shape[0]-c_w+1,s):
        for y in range(0,data.shape[1]-c_h+1,s):
            #data[x:x + c_w, y:y + c_h] = np.uint8(kmeans.cluster_centers_[fit_array[fit_index]], dtype=int).reshape(c_w, c_h)
            data[x:x+c_w,y:y+c_h] = kmeans.cluster_centers_[fit_array[fit_index]].reshape(c_w, c_h)
            fit_index = fit_index+1
    return data

images, labels =load_mnist()#加载数据

if os.path.exists("feature_file.npy") == False:
    print("not found feature file .npy")
else:
    feature = np.load("feature_file.npy").astype('uint8')
    print(feature)

#for i in range(1000):
    #display_image(rebuild_matrix_data(kmeans,mnist_to_matrix(images[10000+i]),7,7,7))
