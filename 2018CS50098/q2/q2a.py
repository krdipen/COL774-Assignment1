import matplotlib
matplotlib.use('Agg')

import numpy as np
from random import gauss
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import sleep
import sys

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
