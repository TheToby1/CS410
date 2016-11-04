'''Visualises the data file for cs410 camera calibration assignment
To run: %run LoadCalibData.py
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

data = np.loadtxt('data.txt')

'''fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(data[:,0], data[:,1], data[:,2],'k.')

fig = plt.figure()
ax = fig.gca()
ax.plot(data[:,3], data[:,4],'r.')'''

plt.show()

P1 = data[:,0:3]
#map(zip, P1)
P1 = np.asarray(P1)
print P1.shape
print P1
P1 = (1
'''
C3 = np.repeat(P1, 2, axis = 0)

C1 = np.zeros((P1.shape[0]*2,P1.shape[1],P1.shape[2]))
C1[::2] = P1

C2 = np.copy(C3)
C2[::2] = np.zeros(P1.shape)

xy = np.repeat(data[:,4],2, axis = 0)
xy[::2] = data[:,3]
xy = xy*-1

C3 = np.asarray(map(np.multiply, C3, xy))

print C1
pts = np.vstack((C1,C2,C3))

print pts
print pts.shape

A = pts.transpose()
tA = pts
print A
print A.shape
print tA
print tA.shape
AtA = tA.transpose().dot(A)
np.linalg.eig(AtA)
d,v = np.linalg.eig(AtA)'''
