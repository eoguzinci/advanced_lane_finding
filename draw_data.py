import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from image_process import undistort
import camera_calibration

def get_curvature(binary_warped, left_fit,right_fit,ploty, left_fitx, right_fitx):
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    h, w = binary_warped.shape[:2]
    car_center = w/2
    left_lane = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    right_lane = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
    lane_center = (right_lane + left_lane) /2
    lane_width = (right_lane - left_lane) * xm_per_pix 
    offset = (car_center - lane_center) * xm_per_pix
    return left_curverad, right_curverad, offset, lane_width

def draw_lane(image, binary_warped, left_fit, right_fit, Minv, mtx, dist):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # recalculation of the pixels in the left and right lanes
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    undist = undistort(image,mtx,dist)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.figure()
    # plt.imshow(result)

    #draw curvature and center offset
    left_curv, right_curv, offset, lane_width = get_curvature(binary_warped, left_fit, right_fit, ploty, left_fitx, right_fitx)
    center_curv = (left_curv+right_curv)/2
    # SANITY CHECK
    err_curv = (left_curv-right_curv)/left_curv
    curve_cond = (left_curv<1000.0 and right_curv<1000.0) 
    curve_simil = (err_curv<0.05) 
    lane_cond = (lane_width<4.0 and lane_width>3.4)
    condition = curve_cond and curve_simil and lane_cond
    if condition==True:
        h = result.shape[0]
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Curve radius: ' + '{:04.2f}'.format(center_curv) + 'm'
        cv2.putText(result, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
        shift = ''
        if offset > 0:
            shift = 'right'
        elif offset < 0:
            shift = 'left'
        text = '{:04.3f}'.format(abs(offset)) + 'm ' + shift + ' of center'
        cv2.putText(result, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
        plt.imshow(result)
    else:
        if curve_cond==False:
            print('Curvature calculated wrong!\nLeft curve: '+'{:04.2f}'.format(left_curv)+', Right curve: '+'{:04.2f}'.format(right_curv))
        if curve_simil==False:
            print('Curvatures not similar!\nError = '+'{:0.3f}'.format(err_curv))
        if lane_cond==False:
            if lane_width>4.0:
                print('Lane width exceeds 4m!'+'{:1.3f}'.format(lane_width))
            if lane_width<3.4:
                print('Lane width even smaller than 3.4m')