#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:26:28 2018

@author: osman
"""

import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import line

from camera_calibration import calibrate_cam
from image_process import undistort, perspective_warp, hls_select, combined_threshold, binary_warp_thresh
from find_lane import blind_lane_find, lane_find
from draw_data import draw_lane, get_curvature

global mtx, dist

# If you run the camera calibration for the first time
# Calibration matrix and distortion coefs
# mtx, dist = calibrate_cam()

# Global variables (just to make the moviepy video annotation work)
with open('calibration.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

plt.figure()
test_image = mpimg.imread('./test_images/test3.jpg')
plt.imshow(test_image)

# Undistort image
test_undistort = undistort(test_image,mtx,dist)

# # Visualize undistortion
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(test_image)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(test_undistort)
# ax2.set_title('Undistorted Image', fontsize=30)

# height and width of the image
h,w = test_undistort.shape[:2]

# define source and destination points for transform
marginx = 300
src = np.float32([(688,450),    # top right
                  (1056,680),    # bottom right
                  (250,680),    # bottom left
                  (592,450)])  # top left
dst = np.float32([(w-marginx,0),
                  (w-marginx,h),
                  (marginx,h),
                  (marginx,0)])
print('Width in pixels')
print(w-2*marginx)

# warp image
test_warp, M, Minv = perspective_warp(test_undistort, src, dst)

# plt.figure()
# plt.imshow(test_warp, cmap='gray')
# plt.savefig('warped test.png')

# # Visualize warped image
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(test_undistort)
# x = [src[0][0],src[1][0],src[2][0],src[3][0],src[0][0]]
# y = [src[0][1],src[1][1],src[2][1],src[3][1],src[0][1]]
# ax1.plot(x, y, color='#ff0000', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
# ax1.set_ylim([h,0])
# ax1.set_xlim([0,w])
# ax1.set_title('Undistorted Image', fontsize=30)
# ax2.imshow(test_warp)
# x = [dst[0][0],dst[1][0],dst[2][0],dst[3][0],dst[0][0]]
# y = [dst[0][1],dst[1][1],dst[2][1],dst[3][1],dst[0][1]]
# ax2.plot(x, y, color='#ff0000', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
# ax2.set_title('Warped Image', fontsize=30)

# # Threshold to Saturation
# thresh_sat = (90, 255)    
# hls_binary = hls_select(test_warp, thresh=thresh_sat)

# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(test_warp)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(hls_binary, cmap='gray')
# ax2.set_title('Thresholded S', fontsize=50)
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

combined_binary = combined_threshold(test_warp)

left_coefs, right_coefs = blind_lane_find(combined_binary)
draw_lane(test_image,combined_binary, left_coefs, right_coefs, Minv, mtx, dist)

new_image = mpimg.imread('./test_images/test4.jpg')
new_binary, Minv = binary_warp_thresh(new_image,mtx,dist)
left_coefs, right_coefs = lane_find(new_binary,left_coefs,right_coefs)
draw_lane(new_image,new_binary, left_coefs, right_coefs, Minv, mtx, dist)

plt.show()