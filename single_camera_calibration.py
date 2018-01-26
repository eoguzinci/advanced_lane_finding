import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb

# Read in a calibration image
img = cv2.imread('./camera_cal/calibration2.jpg')
plt.imshow(img)
img_size = (img.shape[1], img.shape[0])

# termination criteria for cv2.cornerSubPix()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# rows and cols
nx = 9
ny = 6

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world data
imgpoints = [] # 2D points on image plane

# Prepare object points, like (0,0,0),(1,0,0),(2,0,0) ... (7,5,0)
objp = np.zeros((nx*ny,3),np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x,y coordinates

# Convert image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)

# If corners found, add object points, image points
if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    # For more accurate corners
    # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # draw and display the corners
    draw_img =cv2.drawChessboardCorners(img, (nx,ny),corners,ret)
    plt.imshow(draw_img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # refine the camera matrix based on a free scaling parameter using cv2.getOptimalNewCameraMatrix()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    # cv2.imwrite('calibresult.png',dst)
    plt.figure()
    plt.imshow(dst)

    offset = 100 # margins of the unwarped image
    # 4 source points
    src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#    src = np.float32(
#                    [[1160,160],    # top right 
#                    [1013,579],      # bottom right
#                    [270,587],      # bottom left
#                    [153,148]]      # top left
#                )
    # 4 destination points dest = np.float32([[,],[,],[,],[,]])
    dest = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
#    dest = np.float32(
#        [[1200,100],         # top right
#        [1200,650],          # bottom right
#        [100,650],          # bottom left
#        [100,100]]          # top left
#    )

    M = cv2.getPerspectiveTransform(src,dest)
    unwarped = cv2.warpPerspective(dst,M,(dst.shape[1],dst.shape[0]),flags=cv2.INTER_LINEAR)

    plt.figure()
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(unwarped),plt.title('Output')
    plt.show()


# img = mpimg.imread('./camera_cal/calibration1.jpg')
# gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(gray1)
# plt.subplot(1,2,2)
# plt.imshow(gray2)