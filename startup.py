# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:15:44 2020

@author: ADMIN
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

start =  pd.read_csv("C:\\Users\\ADMIN\\Desktop\\chetan\\assignment\\4.multilinear regration\\50_Startups.csv")
start.columns
a = start["R&D Spend"]
b = start["Administration"]
c = start["Marketing Spend"]
d = start["State"]
e = start["Profit"]


plt.hist(start['Profit'])
sns.pairplot(start)
sns.boxenplot(data=start)
sns.hist(data=start)

m1 = smf.ols('e~a+b+c',data = start).fit()
m1.params
m1.summary()#0.951

mb = smf.ols('e~b',data = start).fit()
mb.summary()#0.04

mc = smf.ols('e~c',data = start).fit()
mc.summary()#0.559

mbc = smf.ols('e~b+c',data = start).fit()
mbc.summary()#0.610


import statsmodels.api as sm
sm.graphics.influence_plot(m1)
start_new = start.drop(start.index[[19,45,46,48,49]],axis=0)
a = start_new["R&D Spend"]
b = start_new["Administration"]
c = start_new["Marketing Spend"]
d = start_new["State"]
e = start_new["Profit"]
start_new.corr()

start_new.columns
m_new = smf.ols('e~a+b+c',data = start_new).fit()
m_new.summary()#0.96

mb_new = smf.ols('e~b',data = start_new).fit()
mb_new.summary()

sm.graphics.plot_partregress_grid(m_new)

rsq_a = smf.ols('a~b+c',data=start_new).fit().rsquared  
vif_a = 1/(1-rsq_a) # 16.33

rsq_b = smf.ols('b~a+c',data=start_new).fit().rsquared  
vif_b = 1/(1-rsq_b) # 564.98

rsq_c = smf.ols('c~a+b',data=start_new).fit().rsquared  
vif_c = 1/(1-rsq_c) #  564.84

          # Storing vif values in a data frame
d1 = {'Variables':['a','b','c'],'VIF':[vif_a,vif_b,vif_c,]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

m_final = smf.ols('e~a+b+c',data = start_new).fit()
m_final.summary()

profit_pred =m_final.predict(start_new)
print(m_final.conf_int(0.05))

plt.scatter(start_new.Profit,profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
