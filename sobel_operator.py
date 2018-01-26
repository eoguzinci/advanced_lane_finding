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


sxbinary = sobel_directional(img,0)
sybinary = sobel_directional(img,1)

plt.figure()
plt.subplot(121),plt.imshow(img),plt.title('TEST')
plt.subplot(122),plt.imshow(sxbinary, cmap='gray'),plt.title('Sobelx')
plt.show()

plt.figure()
plt.subplot(121),plt.imshow(img),plt.title('TEST')
plt.subplot(122),plt.imshow(sybinary, cmap='gray'),plt.title('Sobely')
plt.show()

# Run the function
mag_binary = sobel_magnitude(img, sobel_kernel=3, thresh=(30, 100))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#plt.imshow(sxbinary, cmap='gray')