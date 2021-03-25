# importing necessary libraries
from scipy import stats # for calculating mode
import random # for random number generation
import warnings
warnings.filterwarnings("ignore") # ignoring warnings

# delta function or delta metric
def delta(a, b):
    return len([1 for i in range(len(a)) if a[i] != b[i]])

# implementation of k-Mode Clustering
def k_mode_clustering(data, k):
    cluster_centroids = [] # list of cluster centroids
    ind_list = [] # list of data-point indices, selected as cluster centroids
    y = [] # target label after clustering
    flag = 1 # flag variable for checking whether there is any change in the clusters, hence terminating the training process
    iterations = 0 # counting the number of training iterations
    
    # selection of k random cluster centroids among the given data-points
    for i in range(k):
        ind = random.randint(0, len(data) - 1) # selection of a data-point index, for the data-point to be a cluster centroid
        while(ind in ind_list): # if the data-point chosen randomly, is already a selected cluster centroid 
            ind = random.randint(0, len(data) - 1)
        cluster_centroids.append(data[ind]) # updating the list of cluster centroids
        ind_list = [data.index(x) for x in cluster_centroids] # updating the list of data-point indices, selected as cluster centroids
        
    # k-Mode Clustering Algorithm ...
    while(1):
        cost = 0 # variable for calculating total cost for each iteration
        iterations += 1
        for i in range(len(data)):
            dis = [delta(centroid, data[i]) for centroid in cluster_centroids] # delta metric values for k clusters of a data-point
            cost += min(dis) # adding the cost
            if len(y) < len(data): # in case of 1st iteration or 1st pass
                y.append(dis.index(min(dis))) # cluster assignment step
            else: # in case of iterations after the 1st pass
                if y[i] == dis.index(min(dis)): # no change in cluster assignment
                    flag = 0
                else:
                    y[i] = dis.index(min(dis)) # cluster re-assignment
                    flag = 1
                    
        # Displaying the Cost for each iteration
        print('Cost i.e., sum of delta metrics for all data-points for Iteration ' + str(iterations) + ': ', cost)
        if flag == 0: # for all the data-points, there is no change in the clusters, hence, training is terminated
            break
            
        # Cluster Centroid Updation
        for label in range(k):
            data_filter = [data[i] for i in range(len(data)) if y[i] == label] # filtering data-points that are assigned a particular cluster label
            cluster_centroids[label] = list(stats.mode(data_filter, axis = 0)[0][0]) # cluster updation by taking mode
    return y, cluster_centroids # returning cluster labels and cluster centroids