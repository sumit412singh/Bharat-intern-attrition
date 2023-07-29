#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/employee-attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df


# In[ ]:


df.Attrition.replace({'Yes': 1 , 'No': 0},inplace=True)


# In[ ]:


df.drop(columns=['EmployeeCount','StandardHours'],inplace=True)
df.columns


# In[ ]:


cat_col = df.select_dtypes(exclude=np.number).columns
num_col = df.select_dtypes(include=np.number).columns
for i in cat_col:
    print(f'\n========= {i} \n')
    print(df[i].value_counts())


# In[ ]:


encoded_cat_col = pd.get_dummies(df[cat_col], drop_first=True)
final_model = pd.concat([df[num_col],encoded_cat_col],axis=1)
final_model


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


# In[ ]:


x = final_model.drop(columns='Attrition')
y = final_model['Attrition']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)
from sklearn.naive_bayes import GaussianNB


# In[ ]:


model = GaussianNB()


# In[ ]:


model.fit(x_train,y_train)
train_pred = model.predict(x_train)
metrics.confusion_matrix(y_train,train_pred)


# In[ ]:




