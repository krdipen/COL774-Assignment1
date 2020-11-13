import matplotlib
matplotlib.use('Agg')

import numpy as np
import math
import matplotlib.pyplot as plt
import sys

data=np.loadtxt(sys.argv[1]+'/logisticX.csv',delimiter=",")
data=data-np.outer(np.ones(data.shape[0]),np.mean(data,axis=0))
data=data/np.outer(np.ones(data.shape[0]),np.std(data,axis=0))
X=np.column_stack((np.ones(data.shape[0]),data))
y=np.loadtxt(sys.argv[1]+'/logisticY.csv',delimiter=",")

def find_hypo(X,thetha):
    hypo = np.ones(X.shape[0])
    for i in range(X.shape[0]):
        hypo[i] = 1/(1+math.exp(-1*np.matmul(thetha.T,X[i])))
    return hypo

def find_hessian(X,thetha):
    hypo = find_hypo(X,thetha)
    W = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        W[i,i] = hypo[i]*(1-hypo[i])
    return np.matmul(X.T,np.matmul(W,X))

def find_ll(y,X,thetha):
    hypo = find_hypo(X,thetha)
    ll = 0
    for i in range(X.shape[0]):
        ll+=y[i]*math.log(hypo[i])+(1-y[i])*math.log(1-hypo[i])
    return ll

# part1 starts here
eta=0.025
alpha=0.000000001
thetha=np.zeros(X.shape[1])
L_thetha_old=find_ll(y,X,thetha)
while(1):
    thetha=thetha-np.matmul(np.linalg.inv(find_hessian(X,thetha)),np.matmul(X.transpose(),find_hypo(X,thetha)-y))
    L_thetha_new=find_ll(y,X,thetha)
    if abs(L_thetha_new-L_thetha_old)<alpha:
        break
    L_thetha_old=L_thetha_new
# part1 ends here

# part2 starts here
y0_x1 = []
y0_x2 = []
y1_x1 = []
y1_x2 = []
for i in range(X.shape[0]):
    if y[i]:
        y1_x1.append(X[i,1])
        y1_x2.append(X[i,2])
    else:
        y0_x1.append(X[i,1])
        y0_x2.append(X[i,2])
x1 = np.linspace(min(X[:,1]),max(X[:,1]),30)
plt.title('Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(y0_x1,y0_x2,color='tomato',marker='o',label='0')
plt.scatter(y1_x1,y1_x2,color='green',marker='^',label='1')
plt.plot(x1,-1*(thetha[0]+thetha[1]*x1)/thetha[2])
plt.legend()
plt.savefig(sys.argv[2]+'/q3b.png')
# part2 ends here
