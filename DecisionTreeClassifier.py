
# coding: utf-8

# In[3]:

import numpy as np
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)


# In[4]:

importances = clf.feature_importances_
print importances


# In[65]:

import pandas as pd
df=pd.read_csv('C:\dtree.txt', sep=',')


# In[66]:

df.values


# In[67]:

df1 = df.dropna()


# In[68]:

df1.values


# In[69]:

factors = df1.iloc[:, 0:4]


# In[70]:

factors.values


# In[71]:

answer = df1.iloc[:, 4:]


# In[62]:

answer.values


# In[72]:

tr = DecisionTreeClassifier()
tr.fit(factors, answer)


# In[73]:

importances = tr.feature_importances_
print importances


# In[ ]:



