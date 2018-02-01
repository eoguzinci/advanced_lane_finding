import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb

# Read in a test image
img = plt.imread('./test.jpg')
plt.imshow(img)

def sobel_directional(img,direction,thresh_min=20,thresh_max=100):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# determine the direction
	if direction == 0:
		param = np.uint8([1,0])
	else:
		param = np.uint8([0,1])
	
	# Sobel matrix
	sobel = cv2.Sobel(gray, cv2.CV_64F, param[0], param[1])
	
	# Calculate abs(Sobel)
	abs_sobel = np.absolute(sobel)

	# Convert the absolute value image to 8-bit
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	# Create a binary threshold to select pixels based on gradient strength
	sbinary = np.zeros_like(scaled_sobel)
	sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
	return sbinary

def sobel_magnitude(img,sobel_kernel=3,thresh=(20,100)):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	param = np.array([[1,0],[0,1]])
	
	# Sobel matrix
	sobelx = cv2.Sobel(gray, cv2.CV_64F, param[0,0], param[0,1],ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, param[1,0], param[1,1],ksize=sobel_kernel)
	
	# Calculate abs(Sobel)
	abs_sobel = np.sqrt(sobelx**2+sobely**2)

	# Convert the absolute value image to 8-bit
	scale_factor = np.max(abs_sobel)/255 
	scaled_sobel = (abs_sobel/scale_factor).astype(np.uint8)
	# scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	# Create a binary threshold to select pixels based on gradient strength
	sbinary = np.zeros_like(scaled_sobel)
	sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	return sbinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad = np.arctan2(abs_sobely,abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(grad)
    binary_output[(grad >= thresh[0]) & (grad <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output

# Computing Sobel in x and y direction
sxbinary = sobel_directional(img,0)
sybinary = sobel_directional(img,1)

# Absolute value of Sobel operators in x and y direction
mag_binary = sobel_magnitude(img, sobel_kernel=3, thresh=(30, 100))

# Computing the gradient(derivative) of Sobel at each pixel
dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

# Combined thresholds
combined = np.zeros_like(dir_binary)
combined[((sxbinary == 1) & (sybinary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

plt.figure()
plt.subplot(121),plt.imshow(img),plt.title('Test image')
plt.subplot(122),plt.imshow(sxbinary, cmap='gray'),plt.title('Sobelx')

plt.figure()
plt.subplot(121),plt.imshow(img),plt.title('Test Image')
plt.subplot(122),plt.imshow(sybinary, cmap='gray'),plt.title('Sobely')

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#plt.imshow(sxbinary, cmap='gray')

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(dir_binary, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Combined threshold', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
