
# coding: utf-8

# In[2]:

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


# In[3]:

prices = pd.read_csv('C:\close_prices.csv', sep=',')
dj = pd.read_csv('C:\djia_index.csv', sep=',')


# In[10]:

pca = PCA(n_components=10)
X = prices.iloc[:, 1:]


# In[11]:

pca.fit(X)


# In[19]:

i = 0
min_ratio = 0.9
cur_ratio = 0.0
while cur_ratio < min_ratio:
    cur_ratio += pca.explained_variance_ratio_[i]
    i += 1
print i # components number to explain min_ratio of dispersion
    


# In[26]:

X_xfmed = pca.transform(X)
first_comp = []
for i in X_xfmed:
    first_comp.append(i[0])


# In[38]:

# Pearson correlation
np.corrcoef(first_comp, dj.iloc[:, 1:], rowvar=0)[0,1]


# In[56]:

max_id = 0
max_weight = 0.0
for id in range(len(pca.components_[0])):
    cur_weight = pca.components_[0][id]
    if cur_weight > max_weight:
        max_weight = cur_weight
        max_id = id
max_id
list(X.columns.values)[max_id]

