import cv2 as opencv
import numpy as np
from mnist import MNIST
from sklearn.cluster import KMeans


def display_image(data):
    opencv.imshow("image", opencv.resize(data, None, fx=2, fy=2, interpolation=opencv.INTER_CUBIC))
    opencv.waitKey(0)
    opencv.destroyAllWindows()


def save_image(matrix):
    opencv.imwrite("./temp/test",opencv.resize(matrix, None, fx=2, fy=2, interpolation=opencv.INTER_CUBIC))


def kmeans_process():
    feature_matrix = np.load('./temp/feature_7x7x7.npy')
    kmeans = KMeans(n_clusters=4096,init='random',algorithm='full').fit(feature_matrix)
    np.save("./temp/kmeans_center_array",kmeans.cluster_centers_)

#重构数据
def rebuild_matrix_c(feature,f_array,c_width,c_hight,step):
    wh = int(len(f_array) ** 0.5)
    data = np.zeros((wh*c_width,wh*c_hight))
    index=0
    for x in range(0,data.shape[0],step):
        for y in range(0,data.shape[1],step):
            data[x:x + c_width, y:y + c_hight] = feature[f_array[index]].reshape(c_width,c_hight)
            index = index+1
    return data

if __name__=='__main__':
    images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='numpy').load_training()
    feature_matrix = np.load("./temp/kmeans_feature_7x7x7_1138.npy")
    f_array = np.load("./temp/train_feature.npy")
    # print(re_feature_matrix(images[0],rebuild_matrix_c(feature_matrix,f_array[0],7,7,7)))
    print(np.linalg.norm(images[0]-rebuild_matrix_c(feature_matrix,f_array[0],7,7,7).flatten()))