# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:59:10 2019

@author: Kadriye
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#%%
iris=load_iris()
x=iris.data
y=iris.target

x= (x- np.min(x))/(np.max(x)-np.min(x))

#%%train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

#%%knn model 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 3)

#%% K fold  CV  K=10

from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator= knn, X=x_train, y=y_train, cv=10)
print("average accruacy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))


#%%

knn.fit(x_train,y_train)
print("test  accuracy: ",knn.score(x_test,y_test))


#%%Grid search cross validation

from sklearn.model_selection import GridSearchCV

grid= {"n_neighbors ":np.arange(1,50)}
knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn, grid,cv=10)
knn_cv.fit(x,y) 


#%%print hyperparameter KNN algoritmasindaki K değeri

print("tuned hyperparameter K:",knn_cv.best_params_)
print("tuned hyperparameter gore en iyi accuracy (best score):",knn_cv.best_score_)


#%% Grid search CV with logistic regression

x=x[:100,:]
y=y[:100]

from sklearn.linear_model import LogisticRegression

grid= {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}


logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x,y)

print("tuned hyperparameter: (best parameters):",logreg_cv.best_params_)
print("tuned hyperparameter gore en iyi accuracy (best score):",logreg_cv.best_score_)



