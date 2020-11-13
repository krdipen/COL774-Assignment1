import matplotlib
matplotlib.use('Agg')

import numpy as np
from random import gauss
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import sleep
import sys
import time

# part1 starts here
x0 = np.ones(1000000)
x1 = np.ones(1000000)
x2 = np.ones(1000000)
thetha = np.array([3,1,2])
y = np.ones(1000000)
for i in range(1000000):
    x1[i] = gauss(3,2)
    x2[i] = gauss(-1,2)
    y[i] = gauss(np.matmul(thetha.transpose(),np.array([x0[i],x1[i],x2[i]])),math.sqrt(2))
data = np.column_stack((x1,x2))
X = np.column_stack((x0,x1,x2))
np.savetxt(sys.argv[2]+"/sampleX.csv", data, delimiter=",")
np.savetxt(sys.argv[2]+"/sampleY.csv", y, delimiter=",")
# part1 ends here

# part3 starts here
data=np.loadtxt(sys.argv[1]+"/q2test.csv", delimiter=",", skiprows=1)
data_X=np.column_stack((np.ones(data.shape[0]),data[:,0:2]))
data_y=data[:,2]
print(f"cost = {(1/(2*y.shape[0]))*np.matmul((np.matmul(data_X,thetha)-data_y).T,np.matmul(data_X,thetha)-data_y)}")
# part3 ends here

# part2 starts here
start=time.time()
eta=0.001
r=1
alpha=15
k=10000
thetha=np.zeros(X.shape[1])
J_thetha_old=(1/(2*r))*np.matmul((np.matmul(X[0*r:(0+1)*r,:],thetha)-y[0*r:(0+1)*r]).T,np.matmul(X[0*r:(0+1)*r,:],thetha)-y[0*r:(0+1)*r])
thetha_0 = [thetha[0]]
thetha_1 = [thetha[1]]
thetha_2 = [thetha[2]]
J_thetha = [J_thetha_old]
count=0
epochs=0
b=False
while(1):
    for i in range(int(1000000/r)):
        thetha=thetha-(eta/r)*np.matmul(X[i*r:(i+1)*r,:].T,np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r])
        J_thetha_new=(1/(2*r))*np.matmul((np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r]).T,np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r])
        thetha_0.append(thetha[0])
        thetha_1.append(thetha[1])
        thetha_2.append(thetha[2])
        J_thetha.append(J_thetha_new)
        if abs(J_thetha_new-J_thetha_old)<alpha:
            count+=1
            if count == k:
                b=True
                break
        else:
            count=0
        J_thetha_old=J_thetha_new
    epochs+=1
    if b:
        break
stop=time.time()
# part2 ends here

# part3 starts here
print(f"time = {stop-start}")
print(f"epochs  = {epochs}")
print(f"cost = {(1/(2*y.shape[0]))*np.matmul((np.matmul(data_X,thetha)-data_y).T,np.matmul(data_X,thetha)-data_y)}")
# part3 ends here

# part2 starts here
start=time.time()
eta=0.001
r=100
alpha=1
k=10000
thetha=np.zeros(X.shape[1])
J_thetha_old=(1/(2*r))*np.matmul((np.matmul(X[0*r:(0+1)*r,:],thetha)-y[0*r:(0+1)*r]).T,np.matmul(X[0*r:(0+1)*r,:],thetha)-y[0*r:(0+1)*r])
thetha_0 = [thetha[0]]
thetha_1 = [thetha[1]]
thetha_2 = [thetha[2]]
J_thetha = [J_thetha_old]
count=0
epochs=0
b=False
while(1):
    for i in range(int(1000000/r)):
        thetha=thetha-(eta/r)*np.matmul(X[i*r:(i+1)*r,:].T,np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r])
        J_thetha_new=(1/(2*r))*np.matmul((np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r]).T,np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r])
        thetha_0.append(thetha[0])
        thetha_1.append(thetha[1])
        thetha_2.append(thetha[2])
        J_thetha.append(J_thetha_new)
        if abs(J_thetha_new-J_thetha_old)<alpha:
            count+=1
            if count == k:
                b=True
                break
        else:
            count=0
        J_thetha_old=J_thetha_new
    epochs+=1
    if b:
        break
stop=time.time()
# part2 ends here

# part3 starts here
print(f"time = {stop-start}")
print(f"epochs  = {epochs}")
print(f"cost = {(1/(2*y.shape[0]))*np.matmul((np.matmul(data_X,thetha)-data_y).T,np.matmul(data_X,thetha)-data_y)}")
# part3 ends here

# part2 starts here
start=time.time()
eta=0.001
r=10000
alpha=0.1
k=100000
thetha=np.zeros(X.shape[1])
J_thetha_old=(1/(2*r))*np.matmul((np.matmul(X[0*r:(0+1)*r,:],thetha)-y[0*r:(0+1)*r]).T,np.matmul(X[0*r:(0+1)*r,:],thetha)-y[0*r:(0+1)*r])
thetha_0 = [thetha[0]]
thetha_1 = [thetha[1]]
thetha_2 = [thetha[2]]
J_thetha = [J_thetha_old]
count=0
epochs=0
b=False
while(1):
    for i in range(int(1000000/r)):
        thetha=thetha-(eta/r)*np.matmul(X[i*r:(i+1)*r,:].T,np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r])
        J_thetha_new=(1/(2*r))*np.matmul((np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r]).T,np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r])
        thetha_0.append(thetha[0])
        thetha_1.append(thetha[1])
        thetha_2.append(thetha[2])
        J_thetha.append(J_thetha_new)
        if abs(J_thetha_new-J_thetha_old)<alpha:
            count+=1
            if count == k:
                b=True
                break
        else:
            count=0
        J_thetha_old=J_thetha_new
    epochs+=1
    if b:
        break
stop=time.time()
# part2 ends here

# part3 starts here
print(f"time = {stop-start}")
print(f"epochs  = {epochs}")
print(f"cost = {(1/(2*y.shape[0]))*np.matmul((np.matmul(data_X,thetha)-data_y).T,np.matmul(data_X,thetha)-data_y)}")
# part3 ends here

# part2 starts here
start=time.time()
eta=0.1
r=1000000
alpha=0.000001
k=1
thetha=np.zeros(X.shape[1])
J_thetha_old=(1/(2*r))*np.matmul((np.matmul(X[0*r:(0+1)*r,:],thetha)-y[0*r:(0+1)*r]).T,np.matmul(X[0*r:(0+1)*r,:],thetha)-y[0*r:(0+1)*r])
thetha_0 = [thetha[0]]
thetha_1 = [thetha[1]]
thetha_2 = [thetha[2]]
J_thetha = [J_thetha_old]
count=0
epochs=0
b=False
while(1):
    for i in range(int(1000000/r)):
        thetha=thetha-(eta/r)*np.matmul(X[i*r:(i+1)*r,:].T,np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r])
        J_thetha_new=(1/(2*r))*np.matmul((np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r]).T,np.matmul(X[i*r:(i+1)*r,:],thetha)-y[i*r:(i+1)*r])
        thetha_0.append(thetha[0])
        thetha_1.append(thetha[1])
        thetha_2.append(thetha[2])
        J_thetha.append(J_thetha_new)
        if abs(J_thetha_new-J_thetha_old)<alpha:
            count+=1
            if count == k:
                b=True
                break
        else:
            count=0
        J_thetha_old=J_thetha_new
    epochs+=1
    if b:
        break
stop=time.time()
# part2 ends here

# part3 starts here
print(f"time = {stop-start}")
print(f"epochs  = {epochs}")
print(f"cost = {(1/(2*y.shape[0]))*np.matmul((np.matmul(data_X,thetha)-data_y).T,np.matmul(data_X,thetha)-data_y)}")
# part3 ends here
