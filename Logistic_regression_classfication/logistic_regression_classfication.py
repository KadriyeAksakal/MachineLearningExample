# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:30:29 2019

@author: Kadriye
"""

#%%libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#%%read csv 

data=pd.read_csv("data.csv")
print(data.info())

data.drop(["Unnamed: 32","id"],axis=1, inplace = True)
data.diagnosis= [1 if each == "M" else 0 for each in data.diagnosis ]
print(data.info())

y=data.diagnosis.values
x_data = data.drop(["diagnosis"],axis= 1)

#%%Normalization

x= (x_data-np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

#%%train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.2, random_state =42)

x_train= x_train.T
x_test= x_test.T
y_train= y_train.T
y_test= y_test.T
 
print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape) 
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

#%%parameter initialize and sigmaid function
#dimension:30

def initialize_weights_and_bias(dimension):
    w=np.full((dimension,1),0.01)
    b=0.0
    return w,b

#w,b =initialize_weights_and_bias(30)

def sigmoid(z):
    y_head= 1/(1+np.exp(-z))
    return y_head

#sigmoid(0)
#%%sklearn with LR
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))






