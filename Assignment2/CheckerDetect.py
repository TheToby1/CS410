""" Example of using OpenCV API to detect and draw checkerboard pattern"""

import numpy as np
import cv2
from calib import *

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WIDTH = 6
HEIGHT = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)
objp = objp * 100

cap = cv2.VideoCapture(0)
with np.load("antiwarp.npz") as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    
jake1 = cv2.imread("jake.jpg")
jake = cv2.resize(jake1, ((HEIGHT-1)*100,(WIDTH-1)*100))

white1 = cv2.imread("white.jpg")
white = cv2.resize(white1, ((HEIGHT-1)*100,(WIDTH-1)*100))

while cv2.waitKey(1) & 0xFF != ord('q'):
        #capture a frame
        ret, img = cap.read()
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        undst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # Our operations on the frame come here
        gray = cv2.cvtColor(undst, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (HEIGHT,WIDTH),None)
        # If found, add object points, image points (after refining them)
        if ret:
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            #Draw and display the corners
            #cv2.drawChessboardCorners(undst, (HEIGHT,WIDTH), corners,ret)
            rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            undst = draw(undst,corners,imgpts)

            proj = calibrateCamera3D(objp,corners)

            jakewarp = cv2.warpPerspective(jake, proj, (w,h))
            whitewarp = cv2.bitwise_not(cv2.warpPerspective(white, proj, (w,h)));
            undst = cv2.bitwise_and(undst, whitewarp)
            undst += jakewarp  
        	  
        cv2.imshow('img',undst)

# release everything
cap.release()
cv2.destroyAllWindows()
