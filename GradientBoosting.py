
# coding: utf-8

# In[42]:

import numpy as np
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier 
from sklearn.metrics import log_loss


# In[2]:

df = pd.read_csv('C:\\gbm-data.csv', sep=',')
X = df.iloc[:, 1:].values
y = np.ravel(df.iloc[:, :1])


# In[3]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


# In[27]:

clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=0.2)


# In[21]:

rates = [1, 0.5, 0.3, 0.2, 0.1]


# In[28]:

clf.fit(X_train, y_train)


# In[39]:

test_score = np.empty(len(clf.estimators_))
train_score = np.empty(len(clf.estimators_))

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    sig = []
    for el in y_pred:
        sig.append(1.0 / (1.0 + math.exp(-el)))
    test_score[i] = log_loss(y_test, sig)

for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
    sig = []
    for el in y_pred:
        sig.append(1.0 / (1.0 + math.exp(-el)))
    train_score[i] = log_loss(y_train, sig)


# In[40]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.figure()
plt.plot(test_score, 'r', linewidth=2)
plt.plot(train_score, 'g', linewidth=2)
plt.legend(['test', 'train'])


# In[41]:

[ [i,v] for i,v in enumerate(test_score) if v == min(test_score) ]


# In[43]:

clf_forest = RandomForestClassifier(n_estimators=36, random_state=241)


# In[44]:

clf_forest.fit(X_train, y_train)


# In[45]:

log_loss(y_test, clf_forest.predict_proba(X_test))


# In[ ]:



