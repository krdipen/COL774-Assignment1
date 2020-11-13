import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import math
import sys

X=np.loadtxt(sys.argv[1]+'/q4x.dat')
X=X-np.outer(np.ones(X.shape[0]),np.mean(X,axis=0))
X=X/np.outer(np.ones(X.shape[0]),np.std(X,axis=0))
y_file=open(sys.argv[1]+'/q4y.dat',"r")
y_list=[]
m0 = 0
m1 = 0
for i in range(X.shape[0]):
    word = y_file.readline()
    if word == 'Alaska\n':
        y_list.append(0)
        m0+=1
    if word == 'Canada\n':
        y_list.append(1)
        m1+=1
y=np.array(y_list)

# part4 starts here
mew0 = np.zeros((1,X.shape[1]))
mew1 = np.zeros((1,X.shape[1]))
for i in range(X.shape[0]):
    if y[i]:
        mew1 = mew1 + X[i]
    else:
        mew0 = mew0 + X[i]
mew0 = mew0/m0
mew1 = mew1/m1
sigma0 = np.zeros((X.shape[1],X.shape[1]))
sigma1 = np.zeros((X.shape[1],X.shape[1]))
for i in range(X.shape[0]):
    if y[i]:
        sigma1 = sigma1 + np.matmul((X[i]-mew1).T,(X[i]-mew1))
    else:
        sigma0 = sigma0 + np.matmul((X[i]-mew0).T,(X[i]-mew0))
sigma0 = sigma0/m0
sigma1 = sigma1/m1
print(mew0)
print(mew1)
print(sigma0)
print(sigma1)
# part4 ends here
