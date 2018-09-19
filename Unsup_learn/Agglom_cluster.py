# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:13:35 2018

@author: Administrator
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs


X, y = make_blobs(random_state=0,n_samples=12)
print(X,y)
agg = AgglomerativeClustering(n_clusters=3)
assign = agg.fit_predict(X)
print(assign)

import mglearn
#mglearn.discrete_scatter(X[:,0], X[:,1], assign)

import matplotlib.pyplot as plt
#plt.xlabel("f0"); plt.ylabel("f1")


from scipy.cluster.hierarchy import dendrogram, ward
link = ward(X)

dendrogram(link)
ax = plt.gca(); bounds = ax.get_xbound()
ax.plot(bounds, [7.25,7.25], '--', c='k')
ax.plot(bounds, [4,4], '--', c='k')


