import numpy as np
import sort_data as sortd
import rebuild_matrix as rebm
from libKMCUDA import kmeans_cuda

#删除无效值的行
def delete_same_rows(data):
    new_array = [tuple(row) for row in data]
    uniques = np.unique(new_array, axis=0)
    print('delete ', data.shape[0] - uniques.shape[0],"same matrix")
    return uniques

def kmeans_data(i_file,o_file,c_array):
    feature = np.load(i_file).astype('float32')
    print(feature.shape)
    for i in range(len(c_array)):
        #centroids,assignments = kmeans_cuda(feature,c_array[i],init="random",yinyang_t=0,metric="cos",verbosity=1)
        centroids, assignments = kmeans_cuda(feature, c_array[i], init="random", yinyang_t=0, verbosity=1)
        center_feature = delete_same_rows(centroids)
        center_feature,_ = sortd.custum_sort_matrix(center_feature,rule=True) #排序矩阵
        np.save(o_file+str(center_feature.shape[0]),center_feature)
        rebm.rebuild_mnist(o_file+str(center_feature.shape[0])+".npy") #重构数据

if __name__ == '__main__':
    kmeans_data('./temp/feature_7x7x7.npy',"./temp/kmeans_feature_7x7x7_",[512])