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

# part1 starts here
mew0 = np.zeros((1,X.shape[1]))
mew1 = np.zeros((1,X.shape[1]))
for i in range(X.shape[0]):
    if y[i]:
        mew1 = mew1 + X[i]
    else:
        mew0 = mew0 + X[i]
mew0 = mew0/m0
mew1 = mew1/m1
sigma = np.zeros((X.shape[1],X.shape[1]))
for i in range(X.shape[0]):
    if y[i]:
        sigma = sigma + np.matmul((X[i]-mew1).T,(X[i]-mew1))
    else:
        sigma = sigma + np.matmul((X[i]-mew0).T,(X[i]-mew0))
sigma = sigma/X.shape[1]
# part1 ends here

# part2 starts here
y0_x0 = []
y0_x1 = []
y1_x0 = []
y1_x1 = []
for i in range(X.shape[0]):
    if y[i]:
        y1_x0.append(X[i,0])
        y1_x1.append(X[i,1])
    else:
        y0_x0.append(X[i,0])
        y0_x1.append(X[i,1])
plt.title('Decision Boundary')
plt.xlabel('growth ring diameters in fresh water')
plt.ylabel('growth ring diameters in marine water')
plt.scatter(y0_x0,y0_x1,color='tomato',marker='o',label='Alaska')
plt.scatter(y1_x0,y1_x1,color='green',marker='^',label='Canada')
plt.legend()
# part2 ends here

# part3 starts here
x0 = np.outer(np.linspace(min(X[:,0]),max(X[:,0]),30),np.ones(30))
x1 = np.outer(np.ones(30),np.linspace(min(X[:,1]),max(X[:,1]),30))
z = np.outer(np.ones(30),np.ones(30))
for i in range(30):
    for j in range(30):
        x = [x0[i][j],x1[i][j]]
        z[i][j] = np.matmul((x-mew1),np.matmul(np.linalg.inv(sigma),(x-mew1).T)) - np.matmul((x-mew0),np.matmul(np.linalg.inv(sigma),(x-mew0).T))-2*math.log(m1/(m0))
plt.contour(x0,x1,z,colors='m',levels=[0])
plt.savefig(sys.argv[2]+'/q4c.png')
# part3 ends here
