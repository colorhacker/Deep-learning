import cv2 as opencv
import numpy as np
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


if __name__=='__main__':
    #print(np.load("./temp/kmeans_feature_7x7x7_4096.npy").astype('uint8'))
    f = np.load("./temp/kmeans_feature_7x7x7_1138.npy")
    for i in range(f.shape[0]):
        print(f[i])
        display_image(f[i].reshape(7,7))
