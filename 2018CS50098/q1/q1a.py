import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import sleep
import sys

data=np.loadtxt(sys.argv[1]+'/linearX.csv',delimiter=",")
data=np.outer(data,np.ones(1))
data=data-np.outer(np.ones(data.shape[0]),np.mean(data,axis=0))
data=data/np.outer(np.ones(data.shape[0]),np.std(data,axis=0))
X=np.column_stack((np.ones(data.shape[0]),data))
y=np.loadtxt(sys.argv[1]+'/linearY.csv',delimiter=",")

# part1 starts here
eta=0.025
alpha=0.000000001
thetha=np.zeros(X.shape[1])
J_thetha_old=(1/(2*data.shape[0]))*np.matmul((np.matmul(X,thetha)-y).transpose(),np.matmul(X,thetha)-y)
thetha_0 = [thetha[0]]
thetha_1 = [thetha[1]]
J_thetha = [J_thetha_old]
while(1):
    thetha=thetha-(eta/data.shape[0])*np.matmul(X.transpose(),np.matmul(X,thetha)-y)
    J_thetha_new=(1/(2*data.shape[0]))*np.matmul((np.matmul(X,thetha)-y).transpose(),np.matmul(X,thetha)-y)
    thetha_0.append(thetha[0])
    thetha_1.append(thetha[1])
    J_thetha.append(J_thetha_new)
    if abs(J_thetha_new-J_thetha_old)<alpha:
        break
    J_thetha_old=J_thetha_new
print(eta)
print(alpha)
print(thetha)
# part1 ends here
