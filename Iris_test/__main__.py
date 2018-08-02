# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:07:30 2018

@author: Administrator
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    iris_data = load_iris()
    print(iris_data.keys())
    
    print(iris_data['target_names'],iris_data['data'])
    
    X_train, X_test, y_train, y_test = train_test_split(
            iris_data['data'], iris_data['target'],random_state=0)
    
    print(X_train.shape,y_train.shape)
    
    iris_data_df = pd.DataFrame(X_train, columns=iris_data.feature_names)
    scatter = pd.scatter_matrix(iris_data_df, c=y_train,figsize=(15,15),
                                marker='o', hist_kwds={'bins':20},s=60,
                                alpha=0.8)
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    
    y_pred = knn.predict(X_test)
    print("test score is :{:.2f}".format(knn.score(X_test,y_test)))
    