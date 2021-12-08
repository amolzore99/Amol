# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:11:52 2021

@author: User
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("auto-mpg.csv")
dataset.shape

dataset.describe()
#Here the variable horsepower is not there though the horsepower variable is continious 
#but still the describe is not showing values need to check whats wrong
dataset.info()
#By using Domain knowledge we know that  Horsepower is integer type data
#but wet get information from this dataset this is object type 
#we need to convert this column object to interger

dataset.horsepower.unique()
#in this horspoer column ? included ,which consider as object type
dataset['hp']=dataset['horsepower'].replace('?',np.nan)
dataset.hp.unique()
dataset['hp'].isnull().sum()
#from above obsevation total 6 null value
#so need to handle this missing value by ither delete or replace it.
#here I handle this missing value replce central tendancy
dataset.hp.dtype
dataset['hp']=dataset['hp'].astype('float64')
dataset.hp.dtype
dataset['hp']=dataset['hp'].fillna(dataset['hp'].median())
dataset['hp'].isnull().sum()

#check co-relation between this all variables
plt.figure(figsize=(100,100))
corr=dataset.corr()
sns.heatmap(corr,annot=True)
sns.pairplot(dataset)
#by plotting pairplot,mileage have good corelation with cylinder, hp, weight & displacement
#cylinder,weight And hp has -ve coreelation with mpg

#Next will be performing the statistical test to find the significance of variable so that we can reduce no.of variable
df=dataset.drop(['horsepower','car name'],axis=1)
import statsmodels.formula.api as sm
test1=sm .ols('mpg~cylinders+displacement+hp+weight+acceleration+origin',df).fit()
test1.summary()

#as in the above summary the p value of the acc is greater than 0.05 so we can remove the acc variable from the dataset
df=dataset.drop(['horsepower','acceleration','car name'],axis=1)
from sklearn.model_selection import train_test_split
x=df.drop(['mpg'],axis=1).values
y=df.mpg.values
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=0)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
model.score(x_test,y_test)


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#(1) predict the mileage for 8cylinder,410 displacement,4000wt,70model,origin2 & 200hp
model.predict([[5,40,4000,70,2,200]])
#Its predict 13.84 mileage

#(2) predict the mileage for 5cylinder,304 displacement,2100wt,80model,origin3 & 110hp
model.predict([[5,304,2100,80,3,110]])
#Its predict 36.03 mileage

#(3)  predict the mileage for 2cylinder,210 displacement,5000wt,75model,origin3 & 94hp
model.predict([[2,210,5000,75,3,94]])
#Its predict 15.30 mileage





