#!/usr/bin/env python
# coding: utf-8

# # opencv practice

# In[2]:


import cv2 as cv


# In[3]:


#read a image


# In[7]:


import cv2 as cv
img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
cv.imshow('rashmika',img)
cv.waitKey(0)


# # VideoCapture()
# To capture a video, you need to create a VideoCapture object. 
# 
# Its argument can be either the device index or the name of a video file.
# A device index is just the number to specify which camera. Normally one camera will be connected. 
# So I simply pass 0 (or -1).
# 
# You can select the second camera by passing 1 and so on. After that, you can capture frame-by-frame. But at the end, don't forget to release the capture.
# 
# # ret
# ret is a boolean variable that returns true if the frame is available.ret" will obtain return value from getting the camera frame, either true of false.we can change ret name to other name.
# 
# # frame
# "Frame" will get the next frame in the camera (via "capture").
# 
# # capture.read()
# cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
# 
# # imshow()
# cv2.imshow() method is used to display an image in a window. The window automatically fits to the image size.
# 
# Syntax: cv2.imshow(window_name, image)
# Parameters: 
# window_name: A string representing the name of the window in which image to be displayed. 
# image: It is the image that is to be displayed.
# 
# 
# # waitKey(25)
# Playing video from file is the same as capturing it from camera, just change the camera index to a video file name. Also while displaying the frame, use appropriate time for cv.waitKey(). If it is too less, video will be very fast and if it is too high, video will be slow (Well, that is how you can display videos in slow motion). 25 milliseconds will be OK in normal cases.
# 
# First, cv2.waitKey(1) & 0xFF will be executed.
# 
# Doing wait 1 ms for user press.
# If user press, for example, q then waitKey return DECIMAL VALUE of q is 113. In Binary, It is expressed as 0b01110001.
# Next, AND operator is excuted with two inputs are 0b01110001 and 0xFF (0b11111111).
# 0b01110001 AND 0b11111111 = 0b01110001. Exactly result is DECIMAL VALUE of q
# 
# Second, compare value of left expression 0b01110001 with ord('q'). Clearly, these values is the same as another value. And final result is break is invoked.
# 
# # isOpened()
#  
# Check if camera opened successfully or working properly
# or it allready open in background
# 

# #  Read video or Play videos

# In[3]:


import cv2 as cv
capture = cv.VideoCapture('F:\\New Tech\\python\\opencv\\rashmika.mp4')
while True:
    ret,frame = capture.read()
    cv.imshow('rashmika',frame)
    if cv.waitKey(25) == ord('q'): #if cv.waitKey(25) & oxFF== ord('q'):
        break  
        
capture.release() # release video capture
cv.destroyAllWindows() # closeing all the frame window


# # Read  from webcam

# In[2]:


import cv2 as cv
capture = cv.VideoCapture(0)
while True:
    ret,frame = capture.read()
    cv.imshow('Camera',frame)
    if ret == True: #check frame is available or not
        if cv.waitKey(1) == ord('q'): #if cv.waitKey(25) & oxFF== ord('q'):
            break          
capture.release() # release video capture
cv.destroyAllWindows()


# # capture photo or write from webcam 

# In[5]:


import cv2 as cv
capture = cv.VideoCapture(0)
while True:
    ret,frame = capture.read()
    cv.imshow('Camera',frame)
    for i in range(50):
        if cv.waitKey(1) == ord('c'):
            cv.imwrite('F:\\opencv'+str(i)+'.jpg',frame)
    if ret == True: 
        if cv.waitKey(1) == ord('q'): #if cv.waitKey(25) & oxFF== ord('q'):
            break          
capture.release() # release video capture
cv.destroyAllWindows()


# # write video

# In[9]:


import cv2 as cv
capture = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'xvid') #codec
#cv.VideoWriter(filename, fourcc, fps, frameSize)
out = cv.VideoWriter('F:\\opencv.avi', fourcc, 20.0, (640,480))
while True:
    ret,frame = capture.read()
    cv.imshow('Camera',frame)
    if ret == True: #check frame is available or not
        if cv.waitKey(1) == ord('q'): #if cv.waitKey(25) & oxFF== ord('q'):
            break          
capture.release() # release video capture
cv.destroyAllWindows()


# # Basic Function

# In[ ]:


#converting into gray image means BGR TO Gray


# In[4]:


import cv2 as cv
img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('rashmika',imgGray)
cv.waitKey(0)


# In[ ]:


# creating Blur image 


# In[15]:


import cv2 as cv
img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(7,7),0)
cv.imshow('rashmika',imgGray)
cv.imshow('rashmika1',imgBlur)
cv.waitKey(0)


# In[ ]:


#conveting canny image


# In[2]:


import cv2 as cv
img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(7,7),0)
imgCanny = cv.Canny(img,100,100)
cv.imshow('rashmika',imgGray)
cv.imshow('rashmika1',imgBlur)
cv.imshow('rashmika2',imgCanny)
cv.waitKey(0)


# In[ ]:





# # Resizing Frames

# In[ ]:





# In[11]:


import cv2 as cv
img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
resize = cv.resize(img, (200,200)) 
cv.imshow('rashmika',resize)
cv.waitKey(0)


# In[15]:


import cv2 as cv
capture = cv.VideoCapture(0)
while True:
    ret,frame = capture.read()
    resize = cv.resize(frame, (720,650)) 
    cv.imshow('Camera',resize)
    if ret == True: #check frame is available or not
        if cv.waitKey(1) == ord('q'): #if cv.waitKey(25) & oxFF== ord('q'):
            break          
capture.release() # release video capture
cv.destroyAllWindows()


# # Drawing Shapes And Putting Text

# In[16]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500),dtype = 'uint8')
cv.imshow('blank img',blank)
cv.waitKey(0)


# In[18]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)
blank[:] = 0,0,255 #BGR = 3 color
cv.imshow('red',blank)
cv.waitKey(0)


# In[ ]:


#RECTANGLE


# In[12]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)
blank[40:50,60:400] = 0,0,255 #BGR = 3 color
#blank [up]
cv.imshow('red',blank)
cv.waitKey(0)


# In[ ]:


#RECTANGLE 1


# In[18]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)
#cv.rectangle(img,point1,point2,color),thickness = 2)
cv.rectangle(blank,(30,30),(250,250),(0,255,0),thickness = 5)
cv.imshow('green',blank)
cv.waitKey(0)


# In[ ]:


#CIRCLE


# In[2]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)
#cv.circle(image, center_coordinates, radius, color, thickness)
cv.circle(blank,(200,200),50,(0,255,0),thickness=2)
cv.imshow('green',blank)
cv.waitKey(0)


#   #LINE

# In[ ]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)
#cv2.line(image, start_point, end_point, color, thickness)
cv.line(blank,(50,50),(100,50),(0,255,0),thickness=2)
cv.imshow('green',blank)
cv.waitKey(0)


# #PUTTING TEXT

# #cv2.putText(image, text, org, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
# 
# image:  It is the image on which text is to be drawn.
# text:  Text string to be drawn.
# org:  It is the coordinates of the bottom-left corner of the text string 
#       in the image. The coordinates are represented as tuples of two values 
#        i.e. (X coordinate value, Y coordinate value).
# 
# font:  It denotes the font type. Some of font types are FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, , etc.
# 
# fontScale:  Font scale factor that is multiplied by the font-specific base size.
# 
# color: It is the color of text string to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
# 
# thickness: It is the thickness of the line in px.
# 
# lineType:   This is an optional parameter.It gives the type of the line to be used.
# 
# bottomLeftOrigin:  This is an optional parameter. When it is true, the image data origin 
#     is at the bottom-left corner. Otherwise, it is at the top-left corner.

# In[1]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)
cv.putText(blank,'kishan',(100,100),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),thickness=2)
cv.imshow('text',blank)
cv.waitKey(0)


# # Bitwise operation

# In[ ]:


BITWISE_AND


# In[2]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)

rectangle = cv.rectangle(blank.copy(),(150,150),(350,350),(0,255,0),thickness = -1)
cv.imshow('rectangle',rectangle)

circle = cv.circle(blank.copy(),(250,250),200,(0,255,0),thickness=-1)
cv.imshow('circle',circle)

bitwise_and = cv.bitwise_and(rectangle,circle)
cv.imshow('bitwise_and',bitwise_and)

cv.waitKey(0)


# In[ ]:


BITWISE_OR


# In[1]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)

rectangle = cv.rectangle(blank.copy(),(150,150),(350,350),(0,255,0),thickness = -1)
cv.imshow('rectangle',rectangle)

circle = cv.circle(blank.copy(),(250,250),200,(0,255,0),thickness=-1)
cv.imshow('circle',circle)

bitwise_or = cv.bitwise_or(rectangle,circle)
cv.imshow('bitwise_or',bitwise_or)

cv.waitKey(0)


# In[ ]:


#XOR


# In[ ]:


import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
cv.imshow('blank img',blank)

rectangle = cv.rectangle(blank.copy(),(150,150),(350,350),(0,255,0),thickness = -1)
cv.imshow('rectangle',rectangle)

circle = cv.circle(blank.copy(),(250,250),200,(0,255,0),thickness=-1)
cv.imshow('circle',circle)

bitwise_xor = cv.bitwise_xor(rectangle,circle)
cv.imshow('bitwise_xor',bitwise_xor)

cv.waitKey(0)


# # Histogram Computing 

# In[ ]:


#cv2.calcHist(images, channels, mask, histSize, ranges, hist, accumulate)

images : it is the source image of type uint8 or float32 represented as “[img]”.

channels : it is the index of channel for which we calculate histogram. For grayscale image, its value is [0] and
color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.

mask : mask image. To find histogram of full image, it is given as “None”.

histSize : this represents our BIN count. For full scale, we pass [256].

ranges : this is our RANGE. Normally, it is [0,256].


# In[ ]:


https://docs.opencv.org/3.4/d1/db7/tutorial_py_histogram_begins.html


# In[2]:


import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
cv.imshow('rashmika',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

gray_hist = cv.calcHist([gray],[0],None,[256],[0,256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('of pixels')
plt.plot(gray_hist)
plt.xlim(0,256)
plt.show()
cv.waitKey(0)


# # Thresholding/Binarizing Image

# In[ ]:


https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html


# In[ ]:


Syntax: cv2.threshold(source, thresholdValue, maxVal, thresholdingTechnique) 
Parameters: 
-> source: Input Image array (must be in Grayscale). 
-> thresholdValue: Value of Threshold below and above which pixel values will change accordingly. 
-> maxVal: Maximum value that can be assigned to a pixel. 
-> thresholdingTechnique: The type of thresholding to be applied. 


# In[6]:


import cv2 as cv
img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
cv.imshow('rashmika',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('rashmika1',gray)

ret,thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY)
cv.imshow('threshold',thresh)

cv.waitKey(0)


# # face Detection With Haar cascades

# In[ ]:


Haar Cascades

Haar Cascade classifiers are an effective way for object detection.Haar Cascade is a machine learning-based
approach where a lot of positive and negative images are used to train the classifier.

Positive images – These images contain the images which we want our classifier to identify.
Negative Images – Images of everything else, which do not contain the object we want to detect.Requirements


# In[ ]:


cv.detectMultiScale (InputArray image, std::vector< Rect > &objects, double scaleFactor=1.1, int minNeighbors=3, 
                  int flags=0, Size minSize=Size(), Size maxSize=Size())


# In[ ]:


https://docs.opencv.org/master/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498


# In[ ]:


https://github.com/opencv/opencv/tree/master/data/haarcascades


# In[2]:


import cv2 as cv

img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
cv.imshow('rashmika',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('rashmika1',gray)

haar_cascade = cv.CascadeClassifier('F:\\New Tech\\python\\opencv\\haarcascade_frontalface_default.xml')

face = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

for (x,y,w,h) in face:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
cv.imshow('Detect Faces',img)

cv.waitKey(0)


# In[ ]:


# IMAGES


# In[3]:


import cv2 as cv

img = cv.imread('F:\\New Tech\\python\\opencv\\rashmika.jpg')
cv.imshow('rashmika',img)

haar_cascade = cv.CascadeClassifier('F:\\New Tech\\python\\opencv\\haarcascade_frontalface_default.xml')
face = haar_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3)

for (x,y,w,h) in face:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
cv.imshow('Detect Faces',img)

cv.waitKey(0)


# In[ ]:


# WEBCAM VIDEO


# In[1]:


import cv2 as cv
haar_cascade = cv.CascadeClassifier('F:\\New Tech\\python\\opencv\\haarcascade_frontalface_default.xml')
capture = cv.VideoCapture(0)
while True:
    ret,frame = capture.read()
    face = haar_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=3)
    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
    cv.imshow('face detect',frame)
    if ret == True: 
        if cv.waitKey(1) == ord('q'):
            break          
capture.release() 
cv.destroyAllWindows()


# In[ ]:




