
# coding: utf-8

# In[15]:

import pandas as pd
import sklearn.datasets
import numpy as np
import sklearn.svm


# In[16]:

df=pd.read_csv('C:\svm-data.csv', sep=',', header=None)


# In[17]:

X = df.iloc[:, 1:]
y = np.ravel(df.iloc[:, :1])


# In[18]:

clf = sklearn.svm.SVC(kernel='linear', C = 100000, random_state=241)
clf.fit(X, y)


# In[19]:

clf.support_


# In[20]:

clf.support_vectors_


# In[ ]:



