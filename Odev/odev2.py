# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 15:19:22 2019

@author: Kadriye
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data=pd.read_csv("voice.csv")

print(data.info())

data.label = [1 if each == "male" else 0 for each in data.label]
print(data.info())

y=data.label.values
x = data.drop(["label"],axis=1)

#%%train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.2, random_state =42)

x_train= x_train.T
x_test= x_test.T
y_train= y_train.T
y_test= y_test.T

#%%initialize and sigmoid function

def initialize_weights_and_bias(dimension):
    w=np.full((dimension,1),0.01)
    b=0.0
    return w,b

def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head

#%%sklearn with LR

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))



