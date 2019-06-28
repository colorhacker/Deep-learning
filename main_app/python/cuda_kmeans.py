import numpy as np
from libKMCUDA import kmeans_cuda


feature = np.load('./feature/feature_0_59.npy').astype('float32')
print(feature.shape)
#centroids,assignments = kmeans_cuda(feature,255,init="random",yinyang_t=0,metric="cos",verbosity=1)
centroids,assignments = kmeans_cuda(feature,128,init="random",yinyang_t=0,verbosity=1)
np.save("feature_file_L2_128",centroids)

