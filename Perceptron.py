
# coding: utf-8

# In[65]:

import pandas as pd
import numpy as np
import sklearn
import sklearn.cross_validation
import sklearn.datasets
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron


# In[66]:

df_train=pd.read_csv('C:\perceptron-train.csv', sep=',', header=None)
df_test =pd.read_csv('C:\perceptron-test.csv', sep=',', header=None)


# In[67]:

X_train = df_train.iloc[:, 1:]
y_train = np.ravel(df_train.iloc[:, :1])


# In[68]:

X_test = df_test.iloc[:, 1:]
y_test = np.ravel(df_test.iloc[:, :1])


# In[76]:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[77]:

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)


# In[78]:

sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))


# In[79]:

clf_scaled = Perceptron(random_state=241)
clf_scaled.fit(X_train_scaled, y_train)


# In[80]:

sklearn.metrics.accuracy_score(y_test, clf_scaled.predict(X_test_scaled))


# In[81]:

sklearn.metrics.accuracy_score(y_train, clf_scaled.predict(X_train_scaled))


# In[ ]:



