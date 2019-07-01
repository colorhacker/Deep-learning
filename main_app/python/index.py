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
    cv2.imshow("image", cv2.resize(data, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#指定方式返回特征矩阵最接近的矩阵
def re_feature_matrix(center_data,data):
    m_class = 0
    #m_len = np.linalg.norm(data-center_data[0])
    m_len = np.cov(data-center_data[0])
    for i in range(1,center_data.shape[0],1):
        n_len = np.cov(center_data[i]-data)
        #n_len = np.linalg.norm(data-center_data[i])
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
            data[x:x + c_w, y:y + c_h] = re_feature_matrix(k_means,data[x:x + c_w, y:y + c_h].flatten()).reshape(c_w,c_h)
    return data


def get_mnist_loss():
    images, label = load_mnist()
    image = np.array(images,dtype='float32')
    loss=0
    for i in range(len(image)):
        aims = rebuild_matrix_data(feature, image[i].reshape(28, 28), 7, 7, 7)
        _l = np.cov(image[i] - aims.flatten())
        loss = loss + _l
        print(i,_l,label[i])
    return loss/len(image)


#删除无效值的行
def delete_same_rows(data):
    #new_array = [tuple(row) for row in data]
    #uniques = np.unique(new_array, axis=0)
    return np.delete(data, np.where(np.isnan(data))[0], axis=0) #删除nan行

f_files = "feature_file_L2_512.npy"
if os.path.exists(f_files) == False:
    print("not found feature file .npy")
else:
    feature = delete_same_rows(np.load(f_files))
    print("feature:",feature.shape,feature.dtype)
    print(get_mnist_loss())
    #
    # images, label = load_mnist()
    # image = np.array(images,dtype='float32')
    # for i in range(1000):
    #     #display_image(np.uint8(image[i], dtype=int).reshape(28, 28))
    #     display_image(rebuild_matrix_data(feature,image[i].reshape(28, 28),7,7,7))
    #
