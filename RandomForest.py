
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold, cross_val_score


# In[2]:

df = pd.read_csv('C:\\abalone.csv', sep=',')
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = df.iloc[:, :-1]
y = np.ravel(df.iloc[:, -1:])


# In[ ]:

scores = {}
for T in range(1, 51):
    clf = RandomForestRegressor(n_estimators=T, random_state=1)
    validator = KFold(len(X), n_folds=5, shuffle=True, random_state=1)
    clf.fit(X, y)
    scores[T] = np.mean(cross_val_score(estimator = clf, X = X, y = y, cv = validator, scoring = 'r2'))


# In[ ]:

scores


# In[3]:

scores2 = {}
validator = KFold(len(X), shuffle=True, n_folds=5, random_state=1)
for T in range(1, 10):
    clf = RandomForestRegressor(n_estimators=T, random_state=1)
    clf.fit(X, y)
    scores2[T] = np.mean(cross_val_score(estimator = clf, X = X, y = y, cv = validator, scoring = 'r2'))


# In[4]:

scores2


# In[5]:

print range(1, 5)


# In[6]:

clf = RandomForestRegressor(n_estimators=21, random_state=1)
validator = KFold(len(X), n_folds=5, shuffle=True, random_state=1)
clf.fit(X, y)
print np.mean(cross_val_score(estimator = clf, X = X, y = y, cv = validator, scoring = 'r2'))


# In[ ]:



