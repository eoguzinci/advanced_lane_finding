## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image10]: ./figures/histogram.png "Histogram"
[image9]: ./figures/lane_find.png "Lane Find"
[image8]: ./figures/test3.png "Threshold Image"
[image7]: ./figures/camera_calibration.png "FindChessCorners"
[image1]: ./figures/undistort_chess.png "Undistorted Chess"
[image2]: ./figures/undistort_road.png "Road Undistort"
[image3]: ./figures/thresholds.png "Binary Example"
[image4]: ./figures/warped_image.png "Warp Example"
[image5]: ./figures/blind_lane_find.png "Blind Lane Find"
[image6]: ./figures/result.png "Output"
[video1]: ./project_video.mp4 "Video"

## Introduction

In this project, we will find the lanes that the car is commuting on the road and highlight it on each frame of an example video. For this project, I have two main files:
1. `project.py` : which illustrates the how the camera calibration, perspective warping, and line detection on the warped images algorithms detect lanes on single images. This preliminary code is written especially to understand the algorithms and illustrate their use cases with plots and images. 
2. `pipeline.py` : which demonstrates how we can use these algorithms on sequential video frames to build a video which highlights the current lane throughout the video.

The main files use libraries:

* `camera_calibration.py` : which obtains the calibration matrix and radial and tangential distortion coefficients of the camera in use using the chessboard images on a plane.
* `image_process.py` : holds the functions undistort images, applying perspective transform and using a combination of gradient and Sobel operators to detect the lines on a warped image as binaries.
* `find_lane.py` : includes two ways of detecting lane lines. 
    * `blind_lane_find`: detects the line by finding the peaks on the histogram and using them as the starting points or guides of the left and right lane lines. The applying sliding window algorithm to track the lines through the warped image.
    * `lane_find` : does not search the lane lines blind, but uses the previously fitted polynomials representing the lane lines. They search the lane lines close to the previously detected ones. 
* `draw_data.py`: Draws the lane on the undistorted image. In addition to that, the algorithm also displays the curvature of the road and the position of the car in the current lane.
* `line.py` : holds the class that stores the information of the left and right lanes.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `camera_calibration.py` file. The function `calibrate_cam()` is called both in `project.py:25` and `pipeline.py:51` only once to obtain the calibration matrix and the distortion coefficients of the camera.

To make a good calibration of our camera for further image processing, the calibration matrix and the distortion coefficients. For this purpose, a flat object with distinctive shape and skin must be used in order to map the 3D images into 2D, and vice versa. Therefore, a chessboard is used due to its sharp contrast between its patterns and flat geometry which makes it easier to detect the corners. The algorithm starts by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

The first to do is to find the fixed points on the chess board, *the corners* on the images. In the `/camera_cal` folder, there are 20 pictures of a chessboard with 7x10 dimensions, which is actually not a original chess board with 8x8 dimensions. The algorithm `cv2.findChessboardCorners()` is used to identify 6x9 corners of the checkerboard:
```python
ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
```
The `corners` are then appended to imgpoints. Once we find the corners, we can increase their accuracy using `cv2.cornerSubPix()`. We can also draw the pattern using `cv2.drawChessboardCorners()`. Here is the chess board corners that the algorithm finds:
![alt text][image7]

*Note: The images that the corners could not be founnd are the ones that some corners of the chessboard is occluded or not appeared*

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function:
```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
```

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The distortion correction function is stored in `image_process.py`. To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Then, I implemented a perspective transform on the undistorted image. The function for my perspective transform is called `perspective warp` inside `image_process.py`. This function warpes the lane lines, defined by source points, on the road image to destination points whcih constitues a flat plane to study on the road in detail. This procedure can be thought also as a region masking of the source rectangle.

The perspective transform is done by the `cv2.getPerspectiveTransform` function which takes the source points in the input image and the destination points that the source points will be stretched in the transformed image and return the transformation matrix, `M`. We also store the inverse transformation matrix, `Minv`, in case of returning to the original image. Then, by using the transformation matrix as input to the `cv2.warpPerspective` function, we obtain the warped image.

The source and the destination points are selected to fit straight lane lines in a rectangular shape in warped image:

```python
# height and width of the image
h,w = img_undistort.shape[:2]
# define source and destination points for transform
marginx = 300
src = np.float32([(688,450),    # top right
                  (1056,680),   # bottom right
                  (250,680),    # bottom left
                  (592,450)])   # top left
dst = np.float32([(w-marginx,0),
                  (w-marginx,h),
                  (marginx,h),
                  (marginx,0)])
```

As the video has a resolution 720p, this resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 688, 450      | 980, 0        | 
| 1056, 680     | 980, 720      |
| 250, 680      | 300, 720      |
| 592, 450      | 300, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

After perspective warping, it is time to detect the right and the left lane lines in the warped image. There are several ways finding the line pattern on an image, however, we  have used a combination of color and gradient thresholds to generate a binary image. First, we have changed color format of the image from RGB to HLS as the hue, lighting and saturation channels are more distinctive than color channels. The color threshold is also applied to the saturation channel with a threshold from 170 to 255. The gradient threshold is applied to the grayscaled image. I used scaled Sobel operator with a threshold from 20 to 100 for the horizontal gradient thresholds. Then, we unite these binary images obtained after the color and gradient thresholds to established single binary with a combined threshold. You can find the fucntion `combined_threshold` in `image_process.py` to investigate the code in detail. In the figure below, you can find the gradient thresholded binaries in green, and color thresholded binaries in blue on the left. The combined threshold binary that will be used fo the lane detection is the union of both thresholds on the right.

![alt text][image8]
![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The fucntions in the `find_lane.py` fit the lane lines with a 2nd order polynomials. The first approach, `blind_lane_find()`, is to search for the lane lines blind without any guide. First, we compute the histogram to find the starting point at the base of the image, at the bottom of the image. For this we divide the search zones into two to seek the left and right lane, separately. The maximum values on the left and right of the center of the image at the histogram are defined as the base points of our left and right lines. 

![alt text][image10]

Then, we search for the binaries in windows large enough to track the other points of the lines through the top of the images. Using windows provides us to selectively pick only the binaries belong to lane lines other than objects surrounding the car. After finding the binary points belong to left and right lines, we fit these lines with a least squares error by calling:
```
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
``` 
![alt text][image5]

The second approach uses the fits of the previous frame the video as guidelines. It searches only the points within a horizontal margin on the left and right-side from the previous fit. After finding the binaries belong to the lane lines of the new image, we fit the lines using the same formula above.

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After we have the fit for the left and right lane lines, I have implemented this [formula](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) in `get_curvature()` function in `draw_data.py`. However, the obtain value was in pixels. So, we need to convert into meters to obtain the right curvature of the road. According to U.S. regulations the lane width is required to be at least 12 feet or 3.7 meters and the dashed lane lines are 10 feet or 3 meters long each. The width of the lane is 680px in the warped image and we have at least 9 dashed line spacings in our warped figure which can be seen at section 2. So, we need to make the conversion below

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/680 # meters per pixel in x dimension
```

To obtain the position of the car with respect to the center of the lane, the camera is assumed to be centered at the middle of the car. The base values, where `x[0]` of the polynomial for the left and right lines calculated to find the position left and right lines. Then, the center of the lane is calculated by averaging the left and right lines. Finally, we are able to calculate the offset of our vehicle from the lane center by simply subtracting the values. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lane lines and the line highlighted drawn on the warped image of the road. Then, by using the inverse perspective transformation matrix, Minv, we have drawn the highlighted on the undistorted image. Besides, the curvature the road and offset of the vehicle from the center of the lane is also displayed on the top left of the undistorted image. The procedure can be found in the `draw_lane()` function inside `draw_data.py`. Here is an example of the output after lane is highlighted and data is displayed.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the pipeline implemented in this project is very well-considered for ill-conditioned images. Here are some examples:

* The fit of the current image is not directly drawn but we make use a stack of last fits to find and highlight the best fit on the image to smooth the images and prevent wrong fits to be drawn at the first elimination.
* Besides, we check if the width of the lane or curvatures are realistic before adding the current fit to the stack of last fits.  

However, the pipeline might still fail in the challenge videos as there are many other thresholding algorithms. With these thresholding, we aim to obtain the best binary image which highlihts only lane lines on the road. We could have used other color spaces which can detect the white and yellow lane lines better than saturation channel or grayscale image. However, if there is a bad weather condition such as snow, even this solution can't save the whole pipeline and I believe that my lane finding algorithm becomes obslete and needs a better computer vision algorithm. 

Besides, the window margins in the blind and guided searches are pretty wide open that some shadow or another vehicle might cause some noise on the lane line detection.
