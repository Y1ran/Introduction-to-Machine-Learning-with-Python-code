# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:24:58 2018

@author: Administrator
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

import mglearn
import numpy as np

def Linear_reg():
    X, y = mglearn.datasets.load_extended_boston()
    
    X_train, X_test, y_train, y_test = train_test_split(
                X, y ,random_state=0)
    
    lr = LinearRegression().fit(X_train,y_train)
    print("lr.coef: {}, bias: {}".format(lr.coef_, lr.intercept_))
    print("lr.test: {:.2f}".format(lr.score(X_test,y_test)))

def Rigde_reg():
    X, y = mglearn.datasets.load_extended_boston()
    
    X_train, X_test, y_train, y_test = train_test_split(
                X, y ,random_state=0)
    ridge = Ridge(alpha=0.5).fit(X_train,y_train)
    print("lr.test: {:.2f}".format(ridge.score(X_test,y_test)))

def Lasso_reg():
    X, y = mglearn.datasets.load_extended_boston()
    
    X_train, X_test, y_train, y_test = train_test_split(
                X, y ,random_state=0)
    lasso = Lasso(alpha=0.0015, max_iter=100000).fit(X_train,y_train)
    print("lr.test: {:.2f}".format(lasso.score(X_test,y_test)))
    print("the number of features is:" , np.sum(lasso.coef_ != 0))