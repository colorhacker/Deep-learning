import numpy as np
from libKMCUDA import kmeans_cuda


feature = np.load('./feature/feature_0_59.npy').astype('float32')
centroids, assignments = kmeans_cuda(feature,512, verbosity=1)
np.save("feature_file",centroids)

