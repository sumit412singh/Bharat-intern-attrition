#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px


# In[ ]:


df=pd.read_csv('sales-forecasting.csv')


# In[ ]:


df[df.duplicated()]


# In[ ]:


df.isnull().sum()


# In[ ]:


df.isnull().sum().sum()


# In[ ]:


df[df['Postal Code'].isnull()]


# In[ ]:


e=df[df['State']=='Delhi']
e.count()  


# In[ ]:


df['Postal Code']= df['Postal Code'].fillna(5401)


# In[ ]:


df['Order Date']=pd.to_datetime(df['Order Date'],format='%d/%m/%Y') 
df['Ship Date']=pd.to_datetime(df['Ship Date'],format='%d/%m/%Y') 


# In[ ]:


sns.countplot(x='Sales',data=df)
plt.title('Sales')


# In[ ]:


sns.countplot(x='Category',data=df)


# In[ ]:


plt.figure(figsize=(16,6))
sns.countplot(x='Sales',data=df)
plt.xticks(rotation='vertical')


# In[ ]:


top_states=df.groupby('State').sum().sort_values('Sales',ascending=False).head(15)
top_states=top_states[['Sales']].round(2)
top_states.reset_index(inplace=True)
top_states


# In[ ]:


plt.figure(figsize=(16,6))
plt.bar(top_states['State'],top_states['Sales'],color='pink',edgecolor='gray')
plt.title("Top 15 States with Sales ",fontsize=16)
plt.xlabel('State',fontsize=14)
plt.ylabel('Sales',fontsize=14)
plt.xticks(rotation='vertical')

for k,v in top_states['Sales'].items():
    if v> 100000:
        plt.text(k,v-60000,f'${str(v)}',rotation=90)
    else:
         plt.text(k,v-20000,f'${str(v)}',rotation=90)


# In[ ]:


top_cities=df.groupby('City').sum().sort_values('Sales',ascending=False).head(15)
top_cities=top_cities[['Sales']].round(2)
top_cities.reset_index(inplace=True)
top_cities


# In[ ]:


plt.figure(figsize=(16,6))
plt.bar(top_cities['City'],top_cities['Sales'],color='pink',edgecolor='gray')
plt.title("Top 15 Cities with Sales ",fontsize=16)
plt.xlabel('City',fontsize=14)
plt.ylabel('Sales',fontsize=14)
plt.xticks(rotation='vertical')

for k,v in top_cities['Sales'].items():
    if v> 100000:
        plt.text(k,v-60000,f'${str(v)}',rotation=90)
    else:
         plt.text(k,v-20000,f'${str(v)}',rotation=90)


# In[ ]:




