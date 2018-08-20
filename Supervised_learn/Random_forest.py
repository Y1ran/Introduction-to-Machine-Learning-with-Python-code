# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:52:16 2018

@author: Administrator
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def Random_forest():

    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    
    X, y= make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target
                , random_state=42)
    
    clf = RandomForestClassifier(n_estimators=1500,max_features=20, 
                                 random_state=0)
    clf.fit(X_train, y_train)
    
    print("train accuracy: {:.2f}".format(clf.score(X_train, y_train)))
    print("test accuracy: {:.2f}".format(clf.score(X_test, y_test)))