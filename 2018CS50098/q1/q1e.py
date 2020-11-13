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
eta=0.001
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
# part1 ends here

# part5 starts here
radius=max(max(thetha[0]-min(thetha_0),max(thetha_0)-thetha[0]),max(thetha[1]-min(thetha_1),max(thetha_1)-thetha[1]))
x_domain = np.outer(np.linspace(thetha[0]-radius,thetha[0]+radius,30),np.ones(30))
y_domain = np.outer(np.linspace(thetha[1]-radius,thetha[1]+radius,30),np.ones(30)).transpose()
z = np.outer(np.ones(30),np.ones(30))
for i in range(30):
    for j in range(30):
        z[i][j] = (1/(2*data.shape[0]))*np.matmul((np.matmul(X,[x_domain[i][j],y_domain[i][j]])-y).T,np.matmul(X,[x_domain[i][j],y_domain[i][j]])-y)
plt.title('Contours of the Error Function')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.contour(x_domain,y_domain,z,levels=J_thetha.reverse())
for i in range(len(J_thetha)):
    plt.scatter(thetha_0[i], thetha_1[i])
    # plt.pause(0.2)
plt.savefig(sys.argv[2]+'/q1e1.png')
# part5 ends here

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
# part1 ends here

# part5 starts here
radius=max(max(thetha[0]-min(thetha_0),max(thetha_0)-thetha[0]),max(thetha[1]-min(thetha_1),max(thetha_1)-thetha[1]))
x_domain = np.outer(np.linspace(thetha[0]-radius,thetha[0]+radius,30),np.ones(30))
y_domain = np.outer(np.linspace(thetha[1]-radius,thetha[1]+radius,30),np.ones(30)).transpose()
z = np.outer(np.ones(30),np.ones(30))
for i in range(30):
    for j in range(30):
        z[i][j] = (1/(2*data.shape[0]))*np.matmul((np.matmul(X,[x_domain[i][j],y_domain[i][j]])-y).T,np.matmul(X,[x_domain[i][j],y_domain[i][j]])-y)
plt.title('Contours of the Error Function')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.contour(x_domain,y_domain,z,levels=J_thetha.reverse())
for i in range(len(J_thetha)):
    plt.scatter(thetha_0[i], thetha_1[i])
    # plt.pause(0.2)
plt.savefig(sys.argv[2]+'/q1e2.png')
# part5 ends here

# part1 starts here
eta=0.1
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
# part1 ends here

# part5 starts here
radius=max(max(thetha[0]-min(thetha_0),max(thetha_0)-thetha[0]),max(thetha[1]-min(thetha_1),max(thetha_1)-thetha[1]))
x_domain = np.outer(np.linspace(thetha[0]-radius,thetha[0]+radius,30),np.ones(30))
y_domain = np.outer(np.linspace(thetha[1]-radius,thetha[1]+radius,30),np.ones(30)).transpose()
z = np.outer(np.ones(30),np.ones(30))
for i in range(30):
    for j in range(30):
        z[i][j] = (1/(2*data.shape[0]))*np.matmul((np.matmul(X,[x_domain[i][j],y_domain[i][j]])-y).T,np.matmul(X,[x_domain[i][j],y_domain[i][j]])-y)
plt.title('Contours of the Error Function')
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.contour(x_domain,y_domain,z,levels=J_thetha.reverse())
for i in range(len(J_thetha)):
    plt.scatter(thetha_0[i], thetha_1[i])
    # plt.pause(0.2)
plt.savefig(sys.argv[2]+'/q1e3.png')
# part5 ends here
