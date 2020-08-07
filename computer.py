# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:52:08 2020

@author: Chetan
"""
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Computer_Data.csv")
data.columns

sns.boxenplot(data=data)
plt.boxplot(data.price)
data.corr()

labelencoder = LabelEncoder()
data['cd'] = labelencoder.fit_transform(data['cd'])
data['multi'] = labelencoder.fit_transform(data['multi'])
data['premium'] = labelencoder.fit_transform(data['premium'])

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

df_norm = norm_func(data.iloc[:,1:])
df_norm.describe()
data.info()
data.head(5)

x= data[['speed','hd','ram','screen','cd','multi','premium','ads','trend']]
y= data[['price']]

m1 = sm.OLS(y,x).fit()
m1.params
m1.summary()

price_pred =m1.predict(x)
print(m1.conf_int(0.05))

plt.scatter(y,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
plt.plot(y,price_pred,color='black');plt.xlabel('Delivery Time');plt.ylabel('Sorting Time')