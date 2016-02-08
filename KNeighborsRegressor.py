
# coding: utf-8

# In[17]:

import pandas as pd
import numpy as np
import sklearn
import sklearn.cross_validation
import sklearn.datasets
from sklearn.neighbors import KNeighborsRegressor
import operator

df = sklearn.datasets.load_boston()


# In[18]:

X = sklearn.preprocessing.scale(df.data)


# In[19]:

y = sklearn.preprocessing.scale(df.target)


# In[35]:

ps = np.linspace(1, 10, num=200)


# In[36]:

regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p = 2)
regressor.fit(X,y)


# In[37]:

accuracies = {}
for pi in ps:
    validator = sklearn.cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=42)
    classifier = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p = pi)
    classifier.fit(X, y)
    accuracies[pi] = np.mean(sklearn.cross_validation.cross_val_score(estimator = classifier, X = X, y = y, cv = validator, scoring = 'mean_squared_error'))


# In[38]:

print max(accuracies.iteritems(), key=operator.itemgetter(1))[0]


# In[39]:

accuracies[1.0]


# In[40]:

accuracies


# In[ ]:



