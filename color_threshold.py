import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test3.jpg')
thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

fig, ax = plt.subplots(figsize=(24,9))
ax.imshow(image, interpolation='nearest')
plt.tight_layout()

# # Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(gray, cmap='gray')
ax1.set_title('Gray', fontsize=30)
ax2.imshow(binary, cmap='gray')
ax2.set_title('Gray Binary', fontsize=30)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)

R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(R, cmap='gray')
ax1.set_title('Red', fontsize=30)
ax2.imshow(G, cmap='gray')
ax2.set_title('Green', fontsize=30)
ax3.imshow(B, cmap='gray')
ax3.set_title('Blue', fontsize=30)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)

thresh = (200, 255)
binary = np.zeros_like(R)
binary[(R > thresh[0]) & (R <= thresh[1])] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(R, cmap='gray')
ax1.set_title('Red Channel', fontsize=30)
ax2.imshow(binary, cmap='gray')
ax2.set_title('Red Threshold:200', fontsize=30)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)

# Converting to HLS
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(H, cmap='gray')
ax1.set_title('Hue', fontsize=30)
ax2.imshow(L, cmap='gray')
ax2.set_title('Lighting', fontsize=30)
ax3.imshow(S, cmap='gray')
ax3.set_title('Saturation', fontsize=30)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)

# Threshold to Saturaion
thresh = (90, 255)
binary = np.zeros_like(S)
binary[(S > thresh[0]) & (S <= thresh[1])] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(S, cmap='gray')
ax1.set_title('Saturation Channel', fontsize=30)
ax2.imshow(binary, cmap='gray')
ax2.set_title('Saturation Threshold: 90', fontsize=30)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)

# Threshold on Hue
thresh = (15, 100)
binary = np.zeros_like(H)
binary[(H > thresh[0]) & (H <= thresh[1])] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(S, cmap='gray')
ax1.set_title('Hue Channel', fontsize=30)
ax2.imshow(binary, cmap='gray')
ax2.set_title('Hue Threshold: (15,100)', fontsize=30)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.)

plt.show()