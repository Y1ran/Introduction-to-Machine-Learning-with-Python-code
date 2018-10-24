# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:28:52 2018

@author: Administrator
"""

import mglearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_breast_cancer

mglearn.plots.plot_scaling()

def test_prep():
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target
                , random_state=1)
    
    scale = MinMaxScaler()
    scale.fit(X_train)
    
    X_scaled = scale.transform(X_train)
    print(X_scaled)
    
def test_SVC():

    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target
                , random_state=42)
    
    svc = SVC(C=100)
    svc.fit(X_train,y_train)
    
    print(svc.score(X_train,y_train))
    print("the test score is {:.2f}".format(svc.score(X_test
                    ,y_test)))
    scale = MinMaxScaler()
    scale.fit(X_train)
    X_test_scaled = scale.transform(X_test)
    X_scaled = scale.transform(X_train)
    
    svc.fit(X_scaled, y_train)
    print("the test score scaled is {:.2f}".format(svc.score(X_test_scaled
                    ,y_test)))
    
    