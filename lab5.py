import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

#load input data
X = np.loadtxt('data_clustering.txt', delimiter=',')
num_clusters = 5

#plot input data
plt.figure()
plt.scatter(X[:,0], X[:, 1], marker='.', facecolors='none', edgecolors='black', s=80)
x_min, x_max= X[:,0].min()-1,X[:,0].max()+1
y_min, y_max= X[:,1].min()-1,X[:,1].max()+1
plt.title('Input data')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())

#create kmeans object
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

#train the kmeans clustering model
kmeans.fit(X)

#step size of the mesh
step_size = 0.01

#define the grid of points to plot the boundaries
x_min, x_max= X[:,0].min()-1,X[:,0].max()+1
y_min, y_max= X[:,1].min()-1,X[:,1].max()+1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

#predict output labels for all the points on the grid
output = kmeans.predict(np.c_[x_vals.ravel(),y_vals.ravel()])

#plot different regions and colour them
output=output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')

#overlay input points
plt.scatter(X[:,0], X[:,1], marker='.', facecolors='none', edgecolors='blue', s=80)

#plot the centers of the clusters
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='o', s=70, linewidths=3, color='white', zorder=12, facecolors='red')

x_min, x_max= X[:,0].min()-1,X[:,0].max()+1
y_min, y_max= X[:,1].min()-1,X[:,1].max()+1
plt.title('Boundaries of clusters')
plt.xlim(x_min, x_max)