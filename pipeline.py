import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from camera_calibration import calibrate_cam
from image_process import undistort, perspective_warp, hls_select, combined_threshold, binary_warp_thresh
from find_lane import blind_lane_find, lane_find
from draw_data import draw_lane, get_curvature
from line import Line

global mtx, dist, left_line, right_line

def process_image(img):
	new_img = np.copy(img)
	img_binary, Minv = binary_warp_thresh(new_img,mtx,dist)
	if not left_line.detected or not right_line.detected:
		l_fit, r_fit = blind_lane_find(img_binary)
	else:
		l_fit, r_fit = lane_find(img_binary, left_line.best_fit, right_line.best_fit)

	# Sanity check for width
	if l_fit is not None and r_fit is not None:
		# calculate x-intercept (bottom of image, x=image_height) for fits
		h = img.shape[0]
		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/680 # meters per pixel in x dimension
		l_fit_x = (l_fit[0]*h**2 + l_fit[1]*h + l_fit[2])*xm_per_pix
		r_fit_x = (r_fit[0]*h**2 + r_fit[1]*h + r_fit[2])*xm_per_pix
		x_diff = r_fit_x-l_fit_x
		if abs(x_diff-3.7) > 0.5:
			l_fit = None
			r_fit = None
        
		left_line.add_fit(l_fit)
		right_line.add_fit(r_fit)

	# draw the current best fit if it exists
	if left_line.best_fit is not None and right_line.best_fit is not None:
		img_out = draw_lane(new_img, img_binary, left_line.best_fit, right_line.best_fit, Minv, mtx, dist)
	else:
		img_out = new_img

	return img_out

# Calibration matrix and distortion coefs
mtx, dist = calibrate_cam()

left_line = Line()
right_line = Line()

video_input = VideoFileClip('project_video.mp4')
video_output = 'project_video_output.mp4'
processed_video = video_input.fl_image(process_image)
processed_video.write_videofile(video_output, audio=False)
