#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 01:34:04 2017

@author: osman
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read and display the original image
img = mpimg.imread('./calibration_wide/GOPR0032.jpg')

plt.imshow(img)

# source image points
plt.plot(850,320,'.') # top right (+,+)
plt.plot(850,420,'.') # bottom right (+,-)
plt.plot(533,420,'.') # bottom left (-,-)
plt.plot(533,320,'.') # top left (-,+)

# define perspective transform function
def warp(img):
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (img.shape[1], img.shape[0])
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    # 1) Undistort using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    
    if ret == True:
        # If found, add object points, image points
        objpoints.append(objp)
        imgpoints.append(corners)
        # a) draw corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        # four source coordinates
        src = np.float32(
                [[850,320],
    			[850,420],
    			[533,420],
    			[533,320]]
                )
    
        # four desired coordinates
        dst = np.float32(
            	    [[870,240],
    			[870,370],
    			[520,370],
    			[520,240]]
        	)
    
        # Compute the perspective transform, M
        M = cv2.getPerspectiveTransform(src, dst)
    
        # Could compute the inverse also by swapping the input param
        Minv = cv2.getPerspectiveTransform(dst, src)
    
        # Create warped image - uses linear interpolation
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
        return warped

# Get perspective transform
warped_im = warp(img)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

ax1.set_title('Source image')
ax1.imshow(img)
ax2.set_title('Warped image')
ax2.imshow(warped_im)