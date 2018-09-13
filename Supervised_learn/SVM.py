# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 17:57:41 2018

@author: Administrator
"""

from sklearn.svm import LinearSVC, SVC
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def SVC_test():
    from sklearn.datasets import make_blobs 
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    
    mglearn.discrete_scatter(X[:,0],X[:,1], y)
    plt.xlabel("F0"); plt.ylabel("F1")
    
    lin_svm = LinearSVC().fit(X, y)
    svm = SVC(kernel='rbf', C=10,gamma=0.1).fit(X,y)
    mglearn.plots.plot_2d_separator(lin_svm, X)
    mglearn.discrete_scatter(X[:,0],X[:,1], y)
    plt.xlabel("F0"); plt.ylabel("F1")
    
    print(svm.dual_coef_)
    
    fig,axes = plt.subplots(3, 3, figsize=(15,10))
    
    for ax, c in zip(axes, [-1, 0, 3]):
        for a, gamma in zip(ax, range(-1,2)):
            mglearn.plots.plot_svm(log_C=c, log_gamma=gamma, ax=a)
    
    axes[0,0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
        ncol=4, loc=(.9,1.2))

def SVC_fit():
    
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target
                , random_state=42)
    
    svc = SVC()
    svc.fit(X_train,y_train)
    
    print(svc.score(X_train,y_train))
    print(svc.score(X_test
                    ,y_test))
    