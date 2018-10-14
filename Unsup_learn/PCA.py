# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:49:47 2018

@author: Administrator
"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_breast_cancer

def test_PCA():

    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target
                , random_state=42)
    
    svc = SVC(C=100)
    svc.fit(X_train,y_train)
    
    print(svc.score(X_train,y_train))
    print("the test score is {:.2f}".format(svc.score(X_test
                    ,y_test)))
    
    X_tmp = StandardScaler().fit_transform(X_train)
    scale = PCA(n_components=2)
    scale.fit(X_tmp)
    X_test_scaled = scale.transform(X_test)
    X_scaled = scale.transform(X_train)
    
    svc.fit(X_scaled, y_train)
    print("the test score scaled is {:.2f}".format(svc.score(X_test_scaled
                    ,y_test)))
    
    