#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


salary_url = "https://raw.githubusercontent.com/rowialfata/RProject/master/Salary_Data.csv"


# In[3]:


from urllib.request import urlretrieve
urlretrieve(salary_url, 'salary.csv')


# In[4]:


df = pd.read_csv("salary.csv")
df.head()


# In[5]:


df.info()


# In[6]:


X = df.iloc[:, :-1]
y = df.iloc[:, 1]


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# In[8]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
# tambahkan .values agar tidak muncul error message
regressor.fit(X_train.values, y_train)


# In[9]:


y_pred = regressor.predict(X_test.values)


# In[10]:


plt.scatter(X_train, y_train, color="green")
plt.plot(X_train, regressor.predict(X_train.values), color="red")
plt.title("Years Experience vs Salary")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()


# In[11]:


plt.scatter(X_test, y_test, color="green")
plt.plot(X_train, regressor.predict(X_train.values), color="red")
plt.title("Years Experience vs Salary")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()


# In[12]:


salary_pred = regressor.predict([[12]])
print("Total gaji untuk pengalaman tersebut adalah:", salary_pred)

