import numpy as np
from libKMCUDA import kmeans_cuda


feature = np.load('./feature/feature_0_59.npy').astype('float32')
print(feature.shape)
centroids,assignments,metric = kmeans_cuda(feature,512,yinyang_t=0,metric='cos', verbosity=1)
np.save("feature_file_"+metric,centroids)

