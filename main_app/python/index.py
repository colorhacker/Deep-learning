from mnist import MNIST
from PIL import Image
from sklearn.cluster import KMeans
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

def mnsit_feature_data(c_w,c_h,s=1):
    images, labels = load_mnist()
    result = np.empty(shape=[0,c_w*c_h], dtype=int)
    #for i in range(len(images)):
    for i in range(10):
        #print("count:",i)
        data = np.uint8(images[i], dtype=int).reshape(28, 28)
        for x in range(data.shape[0]-c_w+1):
            for y in range(data.shape[1]-c_h+1):
                #print(data[x:x+c_w,y:y+c_h].flatten())
                result = np.row_stack((result, data[x:x+c_w,y:y+c_h].flatten()))
                y = y+s
            x = x+s
    return result

def kmeans_feature(data,n_class):
    kmeans = KMeans(n_clusters=n_class).fit(data)
    #y_kmeans = kmeans.predict(data)
    return kmeans.cluster_centers_ # kmeans.labels_ # kmeans.inertia_

def rebuild_mnist(kmeans,data,c_w,c_h,s=1):
    for x in range(data.shape[0] - c_w + 1):
        for y in range(data.shape[1] - c_h + 1):
            #print(data[x:x+c_w,y:y+c_h].flatten())
            #data[x:x + c_w, y:y + c_h].flatten()
            print(kmeans.predict(data[x:x+c_w,y:y+c_h].flatten()))
            #print(kmeans.fit_predict(data[x:x+c_w,y:y+c_h].flatten()))
            y = y + s
        x = x + s

images, labels =load_mnist()

result = mnsit_feature_data(7,7,1)
kmeans = kmeans_feature(result,16)

rebuild_mnist(kmeans,np.uint8(images[0], dtype=int).reshape(28, 28),7,7,7)
#for i in range(kmeans.shape[0]):
#    display_image(np.uint8(kmeans[i], dtype=int).reshape(7, 7))
