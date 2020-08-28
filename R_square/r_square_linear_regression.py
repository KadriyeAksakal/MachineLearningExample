# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:24:57 2019

@author: Kadriye
"""

import pandas as pd
import matplotlib.pyplot as plt

#import data
df=pd.read_csv("linear-regression-dataset.csv",sep =";")

#plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%%  linear regression

#sklearn library
from sklearn.linear_model import LinearRegression

 #linear regression model
linear_reg = LinearRegression()
 
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)
 
linear_reg.fit(x,y)

y_head= linear_reg.predict(x) #maas
plt.plot(x,y_head,color="red")