
# coding: utf-8

# In[1]:

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


# In[2]:

train = pd.read_csv('C:\salary-train.csv', sep=',')
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True).str.lower()


# In[4]:

test = pd.read_csv('C:\salary-test-mini.csv', sep=',')
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True).str.lower()


# In[5]:

vectorizer = TfidfVectorizer(min_df=5)
X_train_desc = vectorizer.fit_transform(train['FullDescription'])
X_test_desc = vectorizer.transform(test['FullDescription'])


# In[8]:

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)


# In[10]:

enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))


# In[11]:

X_train = hstack([X_train_desc,X_train_categ])
X_test = hstack([X_test_desc,X_test_categ])
y_train = train.SalaryNormalized


# In[12]:

clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)


# In[13]:

for s in clf.predict(X_test):
    print s,


# In[ ]:



