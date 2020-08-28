# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:45:33 2019

@author: Kadriye
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("column_3C_weka.csv")

#df.head()
#df.info()
#df.describe()

#x= df.pelvic_incidence.values
#y= df.sacral_slope.values

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,3].values.reshape(-1,1)

#visualize
plt.scatter(df.pelvic_incidence,df.sacral_slope)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()

#%%linear reggression

from sklearn.linear_model import LinearRegression
reg=LinearRegression()

#predict space
predict_space= np.linspace(min(x), max(x)).reshape(-1,1)

reg.fit(x,y)

#predict
predicted =reg.predict(predict_space)

#r_square
print("R^2 score: ",reg.score(x,y))

#visualize
plt.plot(predict_space, predicted, color="black", linewidth = 3)
plt.scatter(x=x, y=y)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()






