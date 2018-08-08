# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 16:13:07 2018

@author: Administrator
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def GBDT():

    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import GradientBoostingClassifier
    cancer = load_breast_cancer()
    
    X, y= make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target
                , random_state=42)
    
    clf = GradientBoostingClassifier(random_state=0, max_depth=5,
                                     learning_rate=0.8, n_estimators=100)
    clf.fit(X_train, y_train)
    
    print("train accuracy: {:.3f}".format(clf.score(X_train, y_train)))
    print("test accuracy: {:.3f}".format(clf.score(X_test, y_test)))