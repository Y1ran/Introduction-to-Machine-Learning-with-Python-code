# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 19:38:34 2018

@author: Administrator
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def k_test():
    X, y = make_blobs(random_state=1)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    
    print(kmeans.predict(X),kmeans.cluster_centers_)
    import mglearn
    
    mglearn.discrete_scatter(X[:,0], X[:,1], kmeans.labels_,markers='o')
    mglearn.discrete_scatter(kmeans.cluster_centers_[:,0],
                             kmeans.cluster_centers_[:,1],[0,1,2],
                             markers='^', markeredgewidth=2)
    
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    assignments = kmeans.labels_
    
    mglearn.discrete_scatter(X[:,0], X[:,1], assignments, ax=axes[0])
    
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    assignments = kmeans.labels_
    
    mglearn.discrete_scatter(X[:,0], X[:,1], assignments, ax=axes[1])
    
    