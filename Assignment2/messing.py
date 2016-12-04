""" Example of using OpenCV API to detect and draw checkerboard pattern"""

import numpy as np
import cv2
from calib import *

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    for i in range(0,4,2):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[1]),(255),3)
	cv2.line(img, tuple(imgpts[i]), tuple(imgpts[3]),(255),3)
	cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i+4]),(255),3)
	cv2.line(img, tuple(imgpts[i+5]), tuple(imgpts[4]),(255),3)
	cv2.line(img, tuple(imgpts[i+5]), tuple(imgpts[6]),(255),3)
	cv2.line(img, tuple(imgpts[i+5]), tuple(imgpts[i+1]),(255),3)

    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WIDTH = 6
HEIGHT = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)
objp = objp * 100

cube3d = np.zeros((4,3), np.float32)
cube3d[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
cube3d = cube3d * 100

cap = cv2.VideoCapture(0)
with np.load("antiwarp.npz") as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

cheight = (HEIGHT-1)*100
cwidth = (WIDTH-1)*100

jake1 = cv2.imread("jake.jpg")
jake = cv2.resize(jake1, (100,100))

white1 = cv2.imread("white.jpg")
white = cv2.resize(white1, (100,100))

axis = np.float32([[0,0,0], [0,cwidth,0], [cheight,cwidth,0], [cheight,0,0],
                   [0,0,-cwidth],[0,cwidth,-cwidth],[cheight,cwidth,-cwidth],[cheight,0,-cwidth]])


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
            
            rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            undst = draw(undst,corners,imgpts)

            #Draw and display the corners
            #cv2.drawChessboardCorners(undst, (HEIGHT,WIDTH), corners,ret)
            imgpts = np.int32(imgpts).reshape(-1,2)


            cubecorn = np.array((imgpts[5],imgpts[6],imgpts[1],imgpts[2])).reshape(4,1,2)
            proj = calibrateCamera3D(cube3d,cubecorn)

            jakewarp = cv2.warpPerspective(jake, proj, (w,h))
            whitewarp = cv2.bitwise_not(cv2.warpPerspective(white, proj, (w,h)));
            undst = cv2.bitwise_and(undst, whitewarp)
            undst += jakewarp

	    cubecorn = np.array((imgpts[7],imgpts[4],imgpts[3],imgpts[0])).reshape(4,1,2)
            proj = calibrateCamera3D(cube3d,cubecorn)

            jakewarp = cv2.warpPerspective(jake, proj, (w,h))
            whitewarp = cv2.bitwise_not(cv2.warpPerspective(white, proj, (w,h)));
            undst = cv2.bitwise_and(undst, whitewarp)
            undst += jakewarp
            
        	  
        cv2.imshow('img',undst)

# release everything
cap.release()
cv2.destroyAllWindows()
