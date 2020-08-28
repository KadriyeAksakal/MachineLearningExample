# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:04:54 2019

@author: Kadriye
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")

#%%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
#%%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

#%% normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,random_state=1)


#%%#%%decision tree 

from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(x_train , y_train)


print("score: ", dt.score(x_test, y_test))

#%%random forest turorials

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100, random_state= 1)
rf.fit(x_train, y_train)

print("random score: ",rf.score(x_test,y_test))

y_pred= rf.predict(x_test)
y_true=y_test


#%%confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_true, y_pred)


#%% cm vizualization

import seaborn as sns 
f, ax =plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()








