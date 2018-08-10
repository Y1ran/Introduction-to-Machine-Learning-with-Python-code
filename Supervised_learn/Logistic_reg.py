# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 18:32:23 2018

@author: Administrator
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import mglearn

def Linear_reg():
    X, y = mglearn.datasets.make_forge()
    
    for model in [LinearSVC(), LogisticRegression()]:
        clf = model.fit(X, y)
    
    mglearn.plots.plot_linear_svc_regularization()
    
def Cancer_test():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target
                , random_state=42)
    
    for model in [LinearSVC(C=0.002), LogisticRegression
                  (C=100,penalty="l1")]:
        clf = model.fit(X_train, y_train)
        print("lr.train: {:.2f}".format(clf.score(X_train,y_train)))
        print("lr.test: {:.2f}".format(clf.score(X_test,y_test)))