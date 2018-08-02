# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:38:57 2018

@author: Administrator
"""
from load_data import *
from knn import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    cancer, bos = Create_data()
    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target , stratify=cancer.target
                , random_state=0)
    train_acurracy = []
    test_acurracy = []
    
    setting = range(1,10)
    
    for k in setting:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        
        train_acurracy.append(clf.score(X_train, y_train))
        test_acurracy.append(clf.score(X_test, y_test))
    
    plt.plot(setting, train_acurracy,label='train')
    plt.plot(setting, test_acurracy,label='test')
    plt.legend()
    