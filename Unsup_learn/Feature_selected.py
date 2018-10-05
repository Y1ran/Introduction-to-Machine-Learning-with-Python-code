# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 21:04:30 2018

@author: Administrator
"""

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

with open('data.csv') as f:
    x = []
    y = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        if len(lines) > 1:
            x.append(float(lines[0]))
            y.append(float(lines[1]))
            
select = RFECV(RandomForestClassifier(n_estimators=100,random_state=42))

import mglearn
citi = mglearn.datasets.load_citibike()
print(citi.head())

from sklearn.preprocessing import OneHotEncoder

en = OneHotEncoder()

X_indx = en.fit_transform(citi.index).toarray()
print(X_indx)
