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

def display_image(data):
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

#测试重构数据的错误值
def get_mnist_loss(feature,count):
    images, label = load_mnist_training()
    #images, label = load_mnist_testing()
    image = np.array(images,dtype='float32')
    loss = 0
    for i in range(count):
    # for i in range(len(image)):
        aims = rebuild_matrix_data(feature, image[i].reshape(28, 28), 7, 7, 7)
        _l = np.cov(image[i] - aims.flatten())
        loss = loss + _l
        #print(i,_l,label[i])
    return loss/count


#删除无效值的行
def delete_nan_rows(data):
    return np.delete(data, np.where(np.isnan(data))[0], axis=0) #删除nan行

def return_loss(file,set):
    X=[]
    Y=[]
    for i in range(len(set)):
        print(set[i])
        f_files = file+"/feature_file_L2_"+str(set[i])+".npy"
        if os.path.exists(f_files) == False:
            print("not found feature file .npy")
        else:
            feature = delete_nan_rows(np.load(f_files))
            print("feature:", feature.shape, feature.dtype)
            X.append(feature.shape[0])
            Y.append(get_mnist_loss(feature,100))
            print("loss",get_mnist_loss(feature,100))
    np.save(file+"/feature_file_L2_X",X)
    np.save(file+"/feature_file_L2_Y",Y)
    return X,Y

if __name__=='__main__':
    images, label = load_mnist_testing()
    image = np.array(images, dtype='float32')
    # for i in range(image.shape[0]):
    # a = np.array([0.4,0.3,0.2,0.1])
    # b = np.array([1,2,3,4])
    # # c = np.corrcoef(a,b)[0][1]
    # c = np.corrcoef(a,b)
    # print(c)


    # X,Y=return_loss("./kmeans_feature_7x7x1",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
    # X,Y=return_loss("./kmeans_feature_7x7x2",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
    # X,Y=return_loss("./kmeans_feature_7x7x3",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
    # X,Y=return_loss("./kmeans_feature_7x7x4",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
    # X,Y=return_loss("./kmeans_feature_7x7x5",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
    # X,Y=return_loss("./kmeans_feature_7x7x7",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])

    # X1=np.load("./kmeans_feature_7x7x1/feature_file_L2_X.npy")
    # X2=np.load("./kmeans_feature_7x7x2/feature_file_L2_X.npy")
    # X3=np.load("./kmeans_feature_7x7x3/feature_file_L2_X.npy")
    # X4=np.load("./kmeans_feature_7x7x4/feature_file_L2_X.npy")
    # X5=np.load("./kmeans_feature_7x7x5/feature_file_L2_X.npy")
    # X7=np.load("./kmeans_feature_7x7x7/feature_file_L2_X.npy")
    #
    # Y1=np.load("./kmeans_feature_7x7x1/feature_file_L2_Y.npy")
    # Y2=np.load("./kmeans_feature_7x7x2/feature_file_L2_Y.npy")
    # Y3=np.load("./kmeans_feature_7x7x3/feature_file_L2_Y.npy")
    # Y4=np.load("./kmeans_feature_7x7x4/feature_file_L2_Y.npy")
    # Y5=np.load("./kmeans_feature_7x7x5/feature_file_L2_Y.npy")
    # Y7=np.load("./kmeans_feature_7x7x7/feature_file_L2_Y.npy")
    # plt.plot(X1,Y1,label ='7x7x1')
    # plt.plot(X2,Y2,label ='7x7x2')
    # plt.plot(X3,Y3,label ='7x7x3')
    # plt.plot(X4,Y4,label ='7x7x4')
    # plt.plot(X5,Y5,label ='7x7x5')
    # plt.plot(X7,Y7,label ='7x7x7')
    # plt.legend()
    # plt.show()

