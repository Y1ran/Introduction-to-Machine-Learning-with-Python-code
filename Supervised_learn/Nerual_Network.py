# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:25:15 2018

@author: Administrator
"""

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import mglearn

def Muti_Layer_perceptron():
    '''test'''
    display(mglearn.plots.plot_logistic_regression_graph())
    X, y= make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(
                X, y ,random_state=0)
    
    mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train,y_train)
    
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)