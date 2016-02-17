
# coding: utf-8

# In[101]:

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import numpy as np
from operator import itemgetter


# In[20]:

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space'])


# In[21]:

X = newsgroups.data
y = newsgroups.target


# In[22]:

vectorizer = TfidfVectorizer()
X_Tfid = vectorizer.fit_transform(X)


# In[28]:

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X_Tfid, y)


# In[33]:

bestSVC = gs.best_estimator_


# In[35]:

bestSVC.fit(X_Tfid, y)


# In[113]:

arr = np.absolute(bestSVC.coef_).toarray()[0, :]
pairs = zip(range(len(arr)), arr)
arr_sorted = sorted(pairs, key=itemgetter(1), reverse=True)
best_pairs = arr_sorted[:10]


# In[114]:

dic = vectorizer.vocabulary_
words_list = []
ids_list = [pair[0] for pair in best_pairs]
for word, id in dic.iteritems():
    if id in ids_list:
        words_list.append(word)
        
print sorted(words_list)


# In[ ]:



