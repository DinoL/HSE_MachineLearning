
# coding: utf-8

# In[107]:

import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score


# In[108]:

df=pd.read_csv('C:\data-logistic.csv', sep=',', header=None)


# In[109]:

X = df.iloc[:, 1:]
y = np.ravel(df.iloc[:, :1])
l = len(y) # objects count


# In[110]:

k = 0.1       # step size
C = 10        # regularization coefficient
eps = 1e-5    # precision
maxIter = 1e5 # maximal iterations count


# In[111]:

def getWeightChange(w, j, C_reg):
    s = 0
    for i in range(l):
        s += y[i] * X[j][i] * (1.0 - 1.0/(1.0 + math.exp(-y[i]*(w[0]*X[1][i] + w[1]*X[2][i]))))
    return k*s/l - C_reg*k*w[j-1]


# In[112]:

def getWeights(C_reg):
    dw = 2*eps # random number
    iter = 0
    w = [0, 0] # initial weights vector
    while dw > eps and iter < maxIter:
        iter += 1
        dw1 = getWeightChange(w, 1, C_reg)
        dw2 = getWeightChange(w, 2, C_reg)
        dw = max(abs(dw1), abs(dw2))
        w[0] += dw1
        w[1] += dw2
    return w


# In[113]:

def getPredictedClass(w):
    y_predicted = []
    for i in range(l):
        y_predicted.append(1.0/(1.0 + math.exp(-w[0]*X[1][i] - w[1]*X[2][i])))
    return y_predicted


# In[114]:

for C_r in [0, 10]:
    print roc_auc_score(y, getPredictedClass(getWeights(C_r)))

