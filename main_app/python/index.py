from mnist import MNIST
from sklearn.cluster import KMeans,MiniBatchKMeans
from scipy.spatial import distance
import numpy as np
import os, random, base64, cv2

def load_mnist():
    data = MNIST('./python-mnist/data')
    image, label = data.load_training()
    return image, label

def display_image( data):
    #cv2.imshow("image", cv2.resize(data, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC))
    cv2.imshow("image", cv2.resize(data, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#采用cosine返回最小距离的特征
def re_feature_matrix(center_data,data):
    m_class = 0
    m_len = np.linalg.norm(data-center_data[0])
    for i in range(1,center_data.shape[0],1):
        #n_len = 0.5+0.5*np.corrcoef(center_data[i],data)[0][1]
        n_len = np.linalg.norm(data-center_data[i])
        if n_len < m_len:
            m_len = n_len
            m_class = i
    return center_data[m_class]

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
            data[x:x + c_w, y:y + c_h] = re_feature_matrix(k_means,data[x:x + c_w, y:y + c_h].flatten().astype('float32')).reshape(c_w,c_h).astype('uint8')
    return data


def print_acc_loss(data,re_data):
    loss=0
    for i in range(data.shape[0]):
        loss = loss + np.linalg.norm(data - re_data)
    return loss/data.shape[0]

images, labels =load_mnist()#加载数据

def delete_same_rows(data):
    new_array = [tuple(row) for row in data]
    uniques = np.unique(new_array, axis=0)
    return uniques

f_files = "feature_file_L2_512.npy"
if os.path.exists(f_files) == False:
    print("not found feature file .npy")
else:
    feature = np.load(f_files).astype('uint8')
    feature = delete_same_rows(feature).astype('float32')
    print(feature.shape)

#for i in range(1000):
    #display_image(rebuild_matrix_data(feature,np.uint8(images[i+1000], dtype=int).reshape(28, 28),7,7,7))
    #display_image(feature[i].reshape(7, 7))
