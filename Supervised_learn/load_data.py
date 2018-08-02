# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:38:47 2018

@author: Administrator
"""

import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston

def Create_data():
    X, y = mglearn.datasets.make_forge()
    
    #mglearn.discrete_scatter(X[:,0], X[:,1],y)
    plt.legend(["Class 0", "Class 1"], loc=4)    
    
    cancer = load_breast_cancer()
    print("cancer key: \n{}".format(cancer.keys()))
    boston = load_boston()
    return cancer, boston