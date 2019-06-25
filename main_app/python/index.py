from mnist import MNIST
from PIL import Image
from sklearn.cluster import KMeans
from time import strftime, localtime
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os, random, base64, cv2

def load_mnist():
    data = MNIST('./python-mnist/data')
    images, labels = data.load_training()
    return images, labels

def display_image(data):
    #display_image(np.uint8(images[0], dtype=int).reshape(28, 28))
    cv2.imshow('img', data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    result = np.empty(shape=[0,c_w*c_h], dtype=int)
    #for i in range(0,len(images),1):
    for i in range(0, 60, 1):
        print(dt.datetime.now(),i)
        result = np.vstack((result, split_feature_data_c(mnist_to_matrix(images[i]),c_w,c_h,s)))
    return result

#进行矩阵kmeans处理
def cpu_kmeans_feature(data,n_class):
    kmeans = KMeans(n_clusters=n_class).fit(data)
    return kmeans,kmeans.cluster_centers_ # kmeans.labels_ # kmeans.inertia_

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
#result = split_feature_data(7,7,1)#分解特征

print("load data..")
result  = np.loadtxt("./f/feature_0.gz")
print("kmeans data..")
kmeans,cluster_centers = cpu_kmeans_feature(result,255)#进行聚类
print("print data..")
for i in range(1000):
    display_image(rebuild_matrix_data(kmeans,mnist_to_matrix(images[10000+i]),7,7,7))
#np.savetxt('feature.gz', result)
#np.savetxt('feature.csv', result, delimiter = ',')