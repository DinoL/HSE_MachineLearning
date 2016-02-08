
# coding: utf-8

# In[142]:

import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import operator

df=pd.read_csv('C:\wine.data.csv', sep=',', header=None)
classes = np.ravel(df.iloc[:, :1])
features = df.iloc[:, 1:]
features_normalized = sklearn.preprocessing.scale(features)
accuracies = {}
accuracies_n = {}
for k in range(1, 51):
    validator = sklearn.cross_validation.KFold(len(features), n_folds=5, shuffle=True, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(features, classes)
    accuracies[k] = np.mean(sklearn.cross_validation.cross_val_score(estimator = classifier, X = features, y = classes, cv = validator, scoring = 'accuracy'))
max(accuracies.values())


# In[143]:

print max(accuracies.iteritems(), key=operator.itemgetter(1))[0]


# In[144]:

accuracies[1]


# In[138]:

for k in range(1, 51):
    validator = sklearn.cross_validation.KFold(len(features_normalized), n_folds=5, shuffle=True, random_state=42)
    classifier_n = KNeighborsClassifier(n_neighbors=k)
    classifier_n.fit(features_normalized, classes)
    accuracies_n[k] = np.mean(sklearn.cross_validation.cross_val_score(estimator = classifier_n, X = features_normalized, y = classes, cv = validator, scoring = 'accuracy'))


# In[139]:

max(accuracies_n.values())


# In[140]:

print max(accuracies_n.iteritems(), key=operator.itemgetter(1))[0]


# In[141]:

accuracies_n[29]


# In[ ]:



