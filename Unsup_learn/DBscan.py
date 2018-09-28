# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:33:24 2018

@author: Administrator
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs,make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_mutual_info_score,adjusted_rand_score

X, y = make_blobs(random_state=0,n_samples=12)
print(X,y)
db = DBSCAN() ; cluster = db.fit_predict(X)

import mglearn
#mglearn.plots.plot_dbscan()

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
db = DBSCAN(eps=0.5)

scaler = StandardScaler() ; X = scaler.fit_transform(X)
cluster = db.fit_predict(X)

mglearn.discrete_scatter(X[:,0], X[:,1], cluster, s=10)

import matplotlib.pyplot as plt
plt.xlabel("f0"); plt.ylabel("f1")

print(adjusted_mutual_info_score(y, cluster),
      adjusted_rand_score(y, cluster))