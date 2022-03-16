# BaggageAI Internship Task

# Problem Statement:

# Case Study:- Image Processing
- You are given two sets of images:- background and threat objects. Background images are the background x-ray images of baggage that gets generated after passing through a X-ray machine at airport. Threat images are the x-ray images of threats that are prohibited at airport while travelling.

- Your task is to cut the threat objects, scale it down, rotate with 45 degree and paste it into the background images using image processing techniques in python.

- Threat objects should be translucent, means it should not look like that it is cut pasted. It should look like that the threat was already there in the background images. Translucent means the threat objects should have shades of background where it is pasted.

- Threat should not go outside the boundary of the baggage. ** difficult **

- If there is any background of threat objects, then it should not be cut pasted into the background images, which means while cutting the threat objects, the boundary of a threat
object should be tight-bound.

# Solution:

# Libraries Used :

- OpenCV
- numpy
- glob
- os
- matplotlib
- itertools

# Methodology

To start with, we read the threat images, background images using the read_images function. For each threat image, it is first converted to grayscale and then dilated with 5x5 matrix of ones with iteration 2. Thi sis done to smooth out the image since the bright area around the threat image gets dilated around the background.
Next, we create a mask for the threat object using a threshold value for white and the cv2 function inRange(). Then, the threat image is cropped to a square using a threshold value using the form_square() function.
The images are padded dynamically so that when the threat is rotated 45 degrees, the whole threat image is covered and nothing is cut out.
Loop through the background images and find the coordinates of the centre of the largest contour found in the background image using get_xy() function.
Next, we fix the threat image according to the x, y position in background image. Finally we lace the threat in the background image using the place_threat() function.

The saved images are stored in the output folder for future reference.

Documentation:

1. read_images(path): This function reads the .jpg files from a specific location and returns a list of images as numpy array and the number of images read.
2. form_square(image): This function takes in a image(threat, with the background set to black using the inRange() OpenCV function)  and finds the left, right, top, and bottom of the threat object, therby removing the extra background. 
NOTE: The threat object is not guaranteed to be a square.  So this function also checks the image for the height and width of the cropped threat image  and pad black portion in top-buttom of left-right making it a square image.
3. pad_image(image): This function calculates the diagonal length of the image and set the height and width of the image equal to diagonal length.
4. get_xy(background): This function craeates a binary image of the baggage using inRange() function and then inverts it.  Next it finds the contours in the binary image and then the contour with maximum area is selected and the center of the countour is found using moments(). 
5. place_threat(background, threat, x=0, y=0): This function places the threat image in the background image in (x, y) location on the background. Defaults to x=0 and y=0.
