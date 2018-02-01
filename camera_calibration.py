import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in a calibration image
img = mpimg.imread('./camera_cal/calibration01.jpg')
plt.imshow(img)

import glob

# Read in and make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world data
imgpoints = [] # 2D points on image plane

rows = 6
cols = 9

# Prepare object points, like (0,0,0),(1,0,0),(2,0,0) ... (7,5,0)
objp = np.zeros((rows*cols,3),np.float32)
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2) #x,y coordinates

for idx,fname in enumerate(images):
    # read in each image
    img = mpimg.imread(fname)

    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray,(cols,rows),None)

    # If corners found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # draw and display the corners
        img =cv2.drawChessboardCorners(img, (cols,rows),corners,ret)
        plt.imshow(img)
        plt.savefig('chessboard_lines'+str(idx)+'.png')

img = mpimg.imread('./camera_cal/calibration01.jpg')
gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(gray1)
plt.subplot(1,2,2)
plt.imshow(gray2)
