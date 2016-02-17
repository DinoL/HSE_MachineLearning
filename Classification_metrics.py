
# coding: utf-8

# In[29]:

import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve


# In[2]:

df=pd.read_csv('C:\classification.csv', sep=',')


# In[11]:

l = len(df)


# In[14]:

TP = 0
TN = 0
FP = 0
FN = 0
for i in range(l):
    if df['pred'][i] == 1:
        if df['true'][i]  == 1:
            TP += 1
        else:
            FP += 1
    else:
        if df['true'][i]  == 1:
            FN += 1
        else:
            TN += 1


# In[16]:

TP, FP, FN, TN


# In[19]:

acc = accuracy_score(df['true'], df['pred'])
prc = precision_score(df['true'], df['pred'])
rec = recall_score(df['true'], df['pred'])
fsc = f1_score(df['true'], df['pred'])


# In[20]:

acc, prc, rec, fsc


# In[21]:

scores=pd.read_csv('C:\scores.csv', sep=',')


# In[25]:

clf_names = ('score_logreg', 'score_svm', 'score_knn', 'score_tree')


# In[27]:

roc_aucs = []
for name in clf_names:
    roc_aucs.append(roc_auc_score(scores['true'], scores[name]))
[ clf_names[i] for i,v in enumerate(roc_aucs) if v == max(roc_aucs) ]


# In[61]:

max_prec_dic = {}
for name in clf_names:
    PR_curve = precision_recall_curve(scores['true'], scores[name])
    prec = [ PR_curve[0][i] for i,v in enumerate(PR_curve[1]) if v >= 0.7 ]
    max_prec_dic[name] = max(prec)
    
[ k for k,v in max_prec_dic.iteritems() if v == max(max_prec_dic.values()) ]

