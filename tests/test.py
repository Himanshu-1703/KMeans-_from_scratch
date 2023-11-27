from sklearn.datasets import make_blobs
from kmeans import KMeans
import matplotlib.pyplot as plt

# generate the cluster data with 5 clusters

X, y = make_blobs(n_samples=200,n_features=2,centers=5,
           cluster_std=[1,0.9,1.4,1.2,0.7],random_state=40)


# fit the model
kmeans = KMeans(n_clusters=5,max_iter=100,random_state=30)

# do the predictions
pred = kmeans.fit_predict(X=X)

# calculate the final centroid values
centroids = kmeans.centroids

# plot the data
plt.scatter(X[:,0],X[:,1],c=pred)

# plot the centroids
plt.scatter(centroids[:,0],centroids[:,1],s=40,marker='*',facecolor='k',edgecolor='k')
plt.show()