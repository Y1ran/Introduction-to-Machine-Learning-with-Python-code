# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 18:05:44 2018

@author: Administrator
"""

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

iris = load_iris()
log_reg = LogisticRegression()

score = cross_val_score(log_reg, iris.data, iris.target,cv=10)
print("cross-vali score is: {}".format(score.mean()))

import mglearn
#mglearn.plots.plot_stratified_cross_validation()

kfold = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in kfold.split(iris.data, iris.target):
    print(train_index, test_index)
    
from sklearn.svm import SVC

def simple_grid(iris, kfold):
    X_train,X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3,random_state = 0)
    best_score = 0
    para_list = [0.001, 0.01, 0.1, 1, 10]
    for  gamma in para_list:
        for C in para_list:
            svm = SVC(gamma=gamma, C=C)
            #svm.fit(X_train, y_train)
            scores = cross_val_score(svm, iris.data, iris.target,cv=kfold)
            score = scores.mean()
            
            if score > best_score:
                best_score = score
                best_para = {'C':C, 'gamma':gamma}
    print("best score is {:.2f}".format(best_score))
    print("best parameters is {}".format(best_para))
    score = cross_val_score(svm, iris.data, iris.target,cv=kfold)
    
    print("CV-score is {}".format(score.mean(0)))
    return best_para

para = simple_grid(iris, kfold)

para_grid = {"C":[0.001, 0.01, 0.1, 1, 10],
                     'gamma':[0.001, 0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(SVC(), para_grid, cv = kfold)
X_train,X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3,random_state = 0)

grid_search.fit(X_train, y_train)
print("best grid score is {:.2f}".format(grid_search.score(X_test,
              y_test)))

import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
display(results.head())

print(cross_val_score(GridSearchCV(SVC(), para_grid, cv = kfold),
                     X_train,y_train, cv = kfold).mean())
y_pred = grid_search.predict(X_test,y_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))