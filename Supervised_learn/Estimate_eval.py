# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:19:17 2018

@author: Administrator
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
from sklearn.datasets import load_iris

def Uncertainty_eval():
    '''test'''
    X, y= make_circles(noise=0.25, factor=0.5, random_state=1)
    
    y = np.array(["blue", "red"])[y]
    X_train, X_test, y_train, y_test = train_test_split(
                X, y ,random_state=0)
    
    gb = GradientBoostingClassifier(learning_rate=0.01,random_state=0)
    gb.fit(X_train, y_train)
    print("Decusuib functions: \n{}".format(gb.decision_function(X_test)[:6]))
    print(gb.classes_)
    print("Decusuib functions: \n{}".format(gb.predict_proba(X_test)[:6]))

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
                iris.data, iris.target ,random_state=42)
    gb.fit(X_train, y_train)
    print(gb.score(X_test,y_test) ,'\n', np.argmax(gb.predict_proba(X_test),axis=1))