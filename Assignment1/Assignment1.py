import numpy as np
import matplotlib.pyplot as plt

def getP(data):
    arr = data[:,0:3]
    return np.hstack((arr, np.ones((arr.shape[0], 1))))

def calibrateCamera3D(data):
    arr = np.repeat(getP(data), 2, axis=0)
    xs = -data[:,3]
    ys = -data[:,4]
    col_1 = arr.copy()
    col_2 = arr.copy()
    col_3 = arr.copy()

    col_1[1::2] = 0
    col_2[::2] = 0
    col_3[::2] *= np.atleast_2d(xs).transpose()
    col_3[1::2] *= np.atleast_2d(ys).transpose()

    A = np.hstack((col_1,col_2,col_3))

    AtA = (A.transpose()).dot(A)
    eig_val, eig_vect = np.linalg.eig(AtA)
    eig_vect = eig_vect.transpose()

    return np.vstack(np.split(eig_vect[-1],3))

def predPnts(data, P):
    arr = getP(data)
    new = np.apply_along_axis(lambda x: P.dot(x), 1, arr)
    new[:,:] /= np.atleast_2d(new[:,-1]).transpose()
    new = np.delete(new,-1,1)
    return new

def visualiseCameraCalibration3D(data, P):
    new = predPnts(data, P)
    plt.plot(new[:,0],new[:,1], 'r*', data[:,-2], data[:,-1], 'b*')
    plt.show()

def evaluateCameraCalibration3D(data, P):
    new = predPnts(data, P)
    orig = data[:,-2:]
    dist = (new-orig)**2
    dist = dist.sum(axis=-1)
    dist = np.sqrt(dist)
    print "Mean: %s" % (np.mean(dist))
    print "Variance: %s" % (np.var(dist))
    print "Minimum: %s" % (dist.min())
    print "Maximum: %s" % (dist.max())


data = np.loadtxt('data.txt')
P = calibrateCamera3D(data)
print P
visualiseCameraCalibration3D(data, P)
evaluateCameraCalibration3D(data, P)
