import cv2 as opencv
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rebuild_matrix as rebulid

def display_image(data):
    opencv.imshow("image", opencv.resize(data, None, fx=2, fy=2, interpolation=opencv.INTER_CUBIC))
    opencv.waitKey(0)
    opencv.destroyAllWindows()


def save_image(matrix):
    opencv.imwrite("./temp/test",opencv.resize(matrix, None, fx=2, fy=2, interpolation=opencv.INTER_CUBIC))


def kmeans_process(n_class,data):
    kmeans = KMeans(n_clusters=n_class,init='random',algorithm='full').fit(data)
    return kmeans.cluster_centers_

#自定义排序矩阵
def custum_sort_matrix(data):
    data_list = data.tolist()
    return np.array(sorted(data_list, key=lambda x:np.linalg.norm(np.zeros(data.shape[1]) - np.array(x))))


if __name__=='__main__':
    # images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='numpy').load_training()
    # feature_matrix = np.load("./temp/kmeans_feature_7x7x7_1138.npy")
    # f_array = np.load("./temp/train_feature.npy")
    # # print(re_feature_matrix(images[0],rebuild_matrix_c(feature_matrix,f_array[0],7,7,7)))
    # print(np.linalg.norm(images[0]-rebuild_matrix_c(feature_matrix,f_array[0],7,7,7).flatten()))

    # new_feature = custum_sort_matrix(np.load("./temp/kmeans_feature_7x7x7_10240.npy"))
    # new_feature = kmeans_process(16, np.load("./temp/kmeans_feature_7x7x7_10240.npy"))
    # np.save("./temp/kmeans_feature_7x7x7_10240_10",new_feature)

    # new_feature = np.load("./temp/kmeans_feature_7x7x7_10240_10.npy")
    # print(new_feature)
    # for i in range(new_feature.shape[0]):
    #     plt.matshow(new_feature[i].reshape(7,7))
    #     plt.show()

    feature = np.load("./temp/kmeans_feature_7x7x7_10240.npy")
    # feature_ten = np.load("./temp/kmeans_feature_7x7x7_10240_10.npy")
    # train = np.load("./temp/test_feature.npy")
    # images, labels = MNIST('./python-mnist/data', mode='randomly_binarized', return_type='numpy').load_testing()
    # for i in range(10):
    # # for i in range(train.shape[0]):
    #     plt.matshow(images[i].reshape(28,28))
    #     plt.matshow(rebulid.rebuild_matrix_c(feature,train[i],7,7,7))
    #     # plt.matshow(rebuild_matrix_c(feature_ten,train[i],7,7,7))
    #     plt.show()

    data = np.copy(feature[2])
    feature = np.delete(feature, 2,axis = 0)
    print(rebulid.re_feature_matrix(feature,data))

    plt.matshow(data.reshape(7, 7))
    plt.matshow(feature[rebulid.re_feature_matrix(feature,data)].reshape(7, 7))
    plt.show()