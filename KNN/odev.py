# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:20:16 2019

@author: Kadriye
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data=pd.read_csv("column-2C-weka.csv")

data.head()

x_data,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

x= (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

#%%knn model 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("{} nn score: {}".format(3, knn.score(x_test, y_test)))

#%%find k value 
score_list = []
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors =each)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()   


