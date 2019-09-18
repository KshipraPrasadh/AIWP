import numpy as np
import matplotlib.pyplot as plt
from  sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

X = np.loadtxt('data_clustering.txt', delimiter=',')

#Estimate bandwidth of X
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

#cluster data with mean shift
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

#Extract the centre of clusters
cluster_centers= meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

#extimate the number of clusters
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print('\nnumberof clusters in input data =', num_clusters)

#PLot the points and cluster centers
plt.figure()
markers='o*xvs'
for i , marker in zip(range(num_clusters),markers):
    #plot the points that belong to the current cluster
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='violet')
    
    #plot the cluster center
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o', markerfacecolor='cyan', markeredgecolor='blue', markersize=15)
    
plt.title('clusters')
plt.show()

