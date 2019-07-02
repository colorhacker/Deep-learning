import numpy as np
from libKMCUDA import kmeans_cuda


def display_loss(file,c_array):
    feature = np.load(file).astype('float32')
    print(feature.shape)
    for i in range(len(c_array)):
        # centroids,assignments = kmeans_cuda(feature,255,init="random",yinyang_t=0,metric="cos",verbosity=1)
        centroids, assignments = kmeans_cuda(feature, c_array[i], init="random", yinyang_t=0, verbosity=1)
        np.save("./kmeans_feature/feature_file_L2_"+str(c_array[i]), centroids)

display_loss('./split_feature/split_feature_7x7x2.npy',[8,16,32,64,128,256,512,768,1024,2048,4096,10240,20480])