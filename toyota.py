# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:57:01 2020

@author: chetan
"""
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("ToyotaCorolla.csv",engine ='python')

data.columns

sns.boxenplot(data=data)
labelencoder = LabelEncoder()

data['Fuel_Type'] = labelencoder.fit_transform(data['Fuel_Type'])
data.Fuel_Type.head(5)

x = data[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
y = data[['Price']]

data.corr()

model = sm.OLS(y, x).fit()
print(model.summary())


import statsmodels.api as sm
sm.graphics.influence_plot(model)
data_new = data.drop(data.index[[80,960,221]],axis=0) # ,inplace=False)

x = data_new[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
y = data_new[['Price']]

model_new = sm.OLS(y, x).fit()
print(model_new.summary())
sm.graphics.influence_plot(model_new)

prediction = model_new.predict(x)

plt.scatter(y,prediction,c='r')
plt.plot(y,prediction,color='black');plt.xlabel('parameter');plt.ylabel('price')








