import numpy as np
from libKMCUDA import kmeans_cuda


def kmeans_data(i_file,o_file,c_array):
    feature = np.load(i_file).astype('float32')
    print(feature.shape)
    for i in range(len(c_array)):
        # centroids,assignments = kmeans_cuda(feature,255,init="random",yinyang_t=0,metric="cos",verbosity=1)
        centroids, assignments = kmeans_cuda(feature, c_array[i], init="random", yinyang_t=0, verbosity=1)
        np.save(o_file+str(c_array[i]), centroids)

kmeans_data('./split_feature/split_feature_7x7x1_t.npy',"./kmeans_feature_7x7x1/f_t_",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
kmeans_data('./split_feature/split_feature_7x7x2_t.npy',"./kmeans_feature_7x7x2/f_t_",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
kmeans_data('./split_feature/split_feature_7x7x3_t.npy',"./kmeans_feature_7x7x3/f_t_",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
kmeans_data('./split_feature/split_feature_7x7x4_t.npy',"./kmeans_feature_7x7x4/f_t_",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
kmeans_data('./split_feature/split_feature_7x7x5_t.npy',"./kmeans_feature_7x7x5/f_t_",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])
kmeans_data('./split_feature/split_feature_7x7x7_t.npy',"./kmeans_feature_7x7x7/f_t_",[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])