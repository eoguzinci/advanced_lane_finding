import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

def calibrate_cam():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # termination criteria learned from opencv tutorials
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # fig, axs = plt.subplots(5,4, figsize=(16, 11))
    # fig.subplots_adjust(hspace = .2, wspace=.001)
    # axs = axs.ravel()

    # Step through the list and search for chessboard corners
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)

            # this part is taken from the camera calibration tutorial at OpenCV-python website:
            # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
        
            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            # axs[i].axis('off')
            # axs[i].imshow(img)
            
    # Test undistortion on an image
    img = cv2.imread('./camera_cal/calibration01.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "calibration.p", "wb" ) )
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    
    # # Visualize undistortion
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    # f.subplots_adjust(hspace = .2, wspace=.05)
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(dst)
    # ax2.set_title('Undistorted Image', fontsize=30)
    return mtx, dist