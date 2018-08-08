# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 19:18:40 2018

@author: Administrator
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

def Tree_pruning():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target
                , random_state=42)
    
    tree = DecisionTreeClassifier(random_state=0, max_depth=5)
    tree.fit(X_train,y_train)
    print("train accuracy: {:.2f}".format(tree.score(X_train, y_train)))
    print("test accuracy: {:.2f}".format(tree.score(X_test, y_test)))
    print("test accuracy: {:}".format(tree.feature_importances_))

    
    export_graphviz(tree, out_file='tree.dot',class_names=["mal","ben"], feature_names=
                    cancer.feature_names, impurity=False, filled=True)
    import graphviz
    
    with open("tree.dot") as fp:
        dot = fp.read()
    graphviz.Source(dot); display(tree)
    
    