import numpy as np


def calibrateCamera3D(objp,imgp):
    objp = np.delete(objp,2,1)
    arr = np.repeat(objp, 2, axis=0)
    arr = np.hstack((arr, np.ones((arr.shape[0], 1))))
    xs = -imgp[:,0,0]
    ys = -imgp[:,0,1]
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
    arindex = np.argmin(eig_val)

    return np.vstack(np.split(eig_vect[arindex],3))

