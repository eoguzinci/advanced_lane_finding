# Notes

### Offset
We can assume that the camera is placed at the center of the car and we can calculate the offset (eccentrcity between the middle of the car and the middle of the lane) by finding the position of the car w.r.t left and right lanes. 

### Tracking
Create a class Line() and create an object for lines at left and right. When you want to update the curvature, polynomial coefs, and etc. for each line, you can just update the data members of the objects.

### Sanity Check
To check the line that you algorithm found are realistic:
* Check if they have similar curvature
* Check the horizontal distance between them 
* Check if they are parallel

### Look Ahead Filter
If you have already find the lane lines for one video frame, you do not have to blindly search the new one as it is not computationally efficient. But you can search the nonzero values after threshold within the windows created by adding a specific margin to the already fitted polynomial. Then again check if you have good curvature again.

### Reset
If your sanity check reveal that the lane lines you have detected are problematic, you can simply asssume that it is bad or difficult frame of the video. What to in such a case:
1. Retain the line properties of the previos fram and use it in the next one
2. If it is wrong again in the next frame too or you lose the lane lines in the margin. Search the lane lines blindly again by starting algorithm from zero.

### Smoothing
Even everything works fine, your curvature and center measurements can jump off frame by frame. Therefore, it is preferable to smooth the values over the last *n* frames from the we have good measurement. So each time you get a new high-confidence measurement, you append it to the list of recent measurements and then take an average over *n* past measurements to obtain the lane position you want to draw onto the image.

### Drawing
You have warped binary and you have fitted the lines with a polynomial and have arrays `ploty`,`left_fitx` and `right_fix`, which represent the x and y pixel values of the lines. Then you can you project those lines on the original image by using the inverse matrix, `M_inv` in the perspective transform.
