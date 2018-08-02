# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:39:03 2018

@author: Administrator
"""
import mglearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def Knn_clf(k):
    mglearn.plots.plot_knn_classification(n_neighbors=k)
    X, y = mglearn.datasets.make_forge()
    
    X_train, X_test, y_train, y_test = train_test_split(
                X, y ,random_state=0)
    clf = KNeighborsClassifier(n_neighbors=k)
    
    clf.fit(X_train,y_train)
    print("test accuracy: {:.2f}".format(clf.score(X_test, y_test)))
    
def Knn_reg(k):
    mglearn.plots.plot_knn_regression(n_neighbors=k)
    