#!/usr/bin/env python
# coding: utf-8

# # Automobile Domain Project for car mileage prediction

# In[1]:


# import Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


#import Dataset
dataset=pd.read_csv("C:/data analyst/project/ml/Car mileage prediction/auto-mpg.csv")
dataset.shape


# In[8]:


dataset.describe()


# In[9]:


#Here the variable horsepower is not there though the horsepower variable is continious 
#but still the describe is not showing values need to check whats wrong


# In[10]:


dataset.info()


# In[11]:


#By using Domain knowledge we know that  Horsepower is integer type data
#but wet get information from this dataset this is object type 
#we need to convert this column object to interger


# In[12]:


dataset.horsepower.unique()


# In[14]:


#in this horspoer column ? included ,which consider as object type
dataset['hp']=dataset['horsepower'].replace('?',np.nan)
dataset.hp.unique()


# In[15]:


dataset['hp'].isnull().sum()


# In[16]:


#from above obsevation total 6 null value
#so need to handle this missing value by ither delete or replace it.
#here I handle this missing value replce central tendancy


# In[17]:


dataset.hp.dtype


# In[20]:


dataset['hp']=dataset['hp'].astype('float64')
dataset.hp.dtype


# In[21]:



dataset['hp']=dataset['hp'].fillna(dataset['hp'].median())
dataset['hp'].isnull().sum()


# In[25]:


#check co-relation between this all variables
plt.figure(figsize=(100,100))
corr=dataset.corr()
corr


# In[26]:


sns.heatmap(corr,annot=True)
sns.pairplot(dataset)


# In[27]:


#by plotting pairplot,mileage have good corelation with cylinder, hp, weight & displacement
#cylinder,weight And hp has -ve coreelation with mpg


# In[29]:


#Next will be performing the statistical test to find the significance of variable so that we can reduce no.of variable
df=dataset.drop(['horsepower','car name'],axis=1)
df.head(5)


# In[30]:


import statsmodels.formula.api as sm
test1=sm .ols('mpg~cylinders+displacement+hp+weight+acceleration+origin',df).fit()
test1.summary()


# In[31]:


#as in the above summary the p value of the acc is greater than 0.05
#so we can remove the acc variable from the dataset


# In[40]:


df=dataset.drop(['horsepower','acceleration','car name'],axis=1)
df


# In[41]:


#Train and split data
from sklearn.model_selection import train_test_split
x=df.drop(['mpg'],axis=1).values
y=df.mpg.values
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=0)


# In[42]:



from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[44]:


y_pred=model.predict(x_test)
model.score(x_test,y_test)


# In[45]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[48]:


#(1) predict the mileage for 8cylinder,410 displacement,4000wt,70model,origin2 & 200hp
model.predict([[8,40,4000,70,2,200]])
#Its predict 6.2600136 mileage


# In[49]:


#(2) predict the mileage for 5cylinder,304 displacement,2100wt,80model,origin3 & 110hp
model.predict([[5,304,2100,80,3,110]])
#Its predict 36.03 mileage


# In[50]:


#(3)  predict the mileage for 2cylinder,210 displacement,5000wt,75model,origin3 & 94hp
model.predict([[2,210,5000,75,3,94]])
#Its predict 15.30 mileage


# In[ ]:




