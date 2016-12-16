import numpy as np
import cv2
import glob

#Utility for getting camera matrix from assignment1
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
    
#Simple code for capturing images    
def screenshot():

    cap = cv2.VideoCapture(0)
    count = 0
    while cv2.waitKey(1) & 0xFF != ord('q'):
        #capture a frame
        ret, img = cap.read()
        cv2.imshow('img',img)    

        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite("test" + str(count) + ".jpg" ,img)
            count += 1

            
    # release everything
    cap.release()
    cv2.destroyAllWindows()

#get distortion Coefficients
def warp():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob("test/*.jpg")

    for filename in images:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(100)
    cv2.destroyAllWindows()        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1])
            
    img2 = cv2.imread('test1.jpg')
    h,  w = img2.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',dst)
        
    np.savez("antiwarp.npz", mtx=mtx, dist=dist)

#Goal of Assignment
#Project image onto a chessboard.
def projImage():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    WIDTH = 6
    HEIGHT = 9
    reso = 100

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
    objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)
    objp = objp * reso

    cap = cv2.VideoCapture(0)

    with np.load("antiwarp.npz") as X:
        mtx, dist = [X[i] for i in ('mtx','dist')]
        
    jake = cv2.resize(cv2.imread("jake.jpg"), ((HEIGHT-1)*reso,(WIDTH-1)*reso))

    white = cv2.resize(cv2.imread("white.jpg"), ((HEIGHT-1)*reso,(WIDTH-1)*reso))

    while cv2.waitKey(1) & 0xFF != ord('q'):
            #capture a frame
            ret, img = cap.read()
            
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            undst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            gray = cv2.cvtColor(undst, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (HEIGHT,WIDTH))
            # If found, add object points, image points (after refining them)
            if ret:
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                proj = calibrateCamera3D(objp,corners)

                jakewarp = cv2.warpPerspective(jake, proj, (w,h))
                whitewarp = cv2.bitwise_not(cv2.warpPerspective(white, proj, (w,h)));
                undst = cv2.bitwise_and(undst, whitewarp) + jakewarp
            	  
            cv2.imshow('img',undst)

    # release everything
    cap.release()
    cv2.destroyAllWindows()

#utility function for jakecube
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
   
#REALLY bad code, messing around with making a cube and projecting images onto different sides.     
def jakeCube():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    WIDTH = 6
    HEIGHT = 9
    reso = 100

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
    objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)
    objp = objp * reso

    cube3d = np.zeros((4,3), np.float32)
    cube3d[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
    cube3d = cube3d * reso

    cap = cv2.VideoCapture(0)
    with np.load("antiwarp.npz") as X:
        mtx, dist = [X[i] for i in ('mtx','dist')]

    cheight = (HEIGHT-1)*reso
    cwidth = (WIDTH-1)*reso

    jake = cv2.resize(cv2.imread("jake.jpg"), (reso,reso))
    rob = cv2.resize(cv2.imread("rob.jpg"), (reso,reso))
    white = cv2.resize(cv2.imread("white.jpg"), (reso,reso))

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
                imgpts = np.int32(imgpts).reshape(-1,2)

                pts = np.copy(imgpts[:4,1])
                temp = pts[0]
                pts[0] = pts[0]+pts[1]
                pts[1] = pts[1]+pts[2]
                pts[2] = pts[2]+pts[3]
                pts[3] = pts[3]+temp
                order = pts.argsort()
                for i in range(0,4):
                    num1 = order[i]
                    if num1==0:
                        cubecorn = np.array((imgpts[4],imgpts[5],imgpts[0],imgpts[1])).reshape(4,1,2)
                        proj = calibrateCamera3D(cube3d,cubecorn)
                        robwarp = cv2.warpPerspective(rob, proj, (w,h))
                        whitewarp = cv2.bitwise_not(cv2.warpPerspective(white, proj, (w,h)));
                        undst = cv2.bitwise_and(undst, whitewarp)
                        undst += robwarp
                    elif num1==1:
                        cubecorn = np.array((imgpts[5],imgpts[6],imgpts[1],imgpts[2])).reshape(4,1,2)
                        proj = calibrateCamera3D(cube3d,cubecorn)
                        jakewarp = cv2.warpPerspective(jake, proj, (w,h))
                        whitewarp = cv2.bitwise_not(cv2.warpPerspective(white, proj, (w,h)));
                        undst = cv2.bitwise_and(undst, whitewarp)
                        undst += jakewarp
                    elif num1==2:
                        cubecorn = np.array((imgpts[6],imgpts[7],imgpts[2],imgpts[3])).reshape(4,1,2)
                        proj = calibrateCamera3D(cube3d,cubecorn)
                        robwarp = cv2.warpPerspective(rob, proj, (w,h))
                        whitewarp = cv2.bitwise_not(cv2.warpPerspective(white, proj, (w,h)));
                        undst = cv2.bitwise_and(undst, whitewarp)
                        undst += robwarp
                    elif num1==3:
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

print "******Press q to see actual assignment*******"
jakeCube()
projImage()
