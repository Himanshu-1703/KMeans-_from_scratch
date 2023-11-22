import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    
    def __init__(self,n_clusters=2,max_iter=100,random_state=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        

    def __cluster_initialize(self,X):
        np.random.seed(self.random_state)
        # index values of X for random selection
        index_values = np.random.randint(0,X.shape[0],size=self.n_clusters)
        # row sampling from X
        sample_rows = X[index_values,:]
        # assign the centroids
        self.centroids = sample_rows

    def __euclidean_distance(self,X,centroid):
        distance = np.square(X-centroid)
        summed_distance = np.sum(distance,axis=1)
        return np.sqrt(summed_distance)

    def __cluster_centroids(self,X,predictions):
        # empty array for each cluster centroid
        cluster_centers = []
        # loop n times for n clusters
        for i in range(self.n_clusters):
            # filter the data based on clusters
            X_filt = X[predictions == i,:]
            # take the mean of cluster data points
            centroid_point = np.mean(X_filt,axis=0)
            cluster_centers.append(centroid_point)
        # make a numpy array of cluster centers
        cluster_centers = np.array(cluster_centers)
        # assign the new centroid values
        self.centroids = cluster_centers


    def fit_predict(self,X):
        # initialize the centroids
        self.__cluster_initialize(X)

        for n in range(self.max_iter):
    
            # for each centroid(value of n_clusters)
            distances = []
            # loop and calculate distance for each cluster
            for i in range(self.n_clusters):
                # calculate the distance for each data point
                centroid_distance = self.__euclidean_distance(X,self.centroids[i,:])
                distances.append(centroid_distance)

            # make an array of all the distance values
            distances = np.array(distances)
            # do the predictions and assign the clusters based on minimum distances
            predictions = np.argmin(distances,axis=0)

            # assign new centroid values based on new clusters
            old_centroids = self.centroids
            self.__cluster_centroids(X,predictions=predictions)
       
            # condition to check the array values elementwise
            if np.array_equal(a1=old_centroids,a2=self.centroids):
                break

        return predictions