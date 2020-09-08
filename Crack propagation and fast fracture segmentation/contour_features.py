import numpy as np
import cv2 as cv

img = cv.imread("images for contours features/2110817386.jpg", 0)
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)

#cnt = contours[0]
#M = cv.moments(cnt)
#print( M )

temp_len = 0
maxContour = contours[0]
#find the largest contours
for contour in contours:
    lengthContour = len(contour)
    if(lengthContour >= temp_len):
        temp_len = lengthContour
        maxContour = contour

cnt = maxContour
M = cv.moments(cnt)
print( M )

cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

#Contour Area
area = cv.contourArea(cnt)

#Contour Perimeter
perimeter = cv.arcLength(cnt, True)

#Contour approximation
epsilon = 0.1*cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)

#Bounding Rectangle
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img, (x,y),(x+w, y+h),(0,255,0),2)

#Rotated Rectangle
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img, [box], 0, (0,0,255),2)

#Minimum Enclosing Circle
(x,y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv.circle(img, center, radius, (0,255,0),2)

################################################
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#img = cv.imread("images for contours features/2110817386.jpg")
#img = cv.imread("images for contours features/threshold 127 full.png")
img = cv.imread("images for contours features/type_2_6.jpg")

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#threshold the image
ret, thresh = cv.threshold(imgray, 127, 255, 0)
#find the contours
#contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.RETR_LIST)
#draw the image by contours
cv.drawContours(img, contours, -1, (0,0,0),3)

#show the images
plt.imshow(img, 'gray')
plt.show()

############################################
#find contours by active contour
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
img = cv.imread("images for contours features/2110817386.jpg")
#"images for contours features/active_contour.png"
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

s = np.linspace(0, 2*np.pi, 1000)
r = 550 + 550*np.sin(s)
c = 950 + 550*np.cos(s)
init = np.array([r, c]).T

snake = active_contour(gaussian(imgray, 3), init, alpha=0.015, beta=10, gamma=0.001,
                       coordinates='rc')
snake2 = snake.reshape((-1,1,2))
snake2 = np.array(snake2, np.int32)
tempSnake = np.copy(snake2[:,:,1])
snake2[:,:,1] = snake2[:,:,0]
snake2[:,:,0] = tempSnake

#plot the images
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(imgray, cmap=plt.cm.gray)
#ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1],img.shape[0],0])

#write image with points
pts = np.array(snake,np.int32)
pts = pts.reshape((-1,1,2))
tempPts = np.copy(pts[:,:,1])
pts[:,:,1] = pts[:,:,0]
pts[:,:,0] = tempPts

#img_contours = cv.polylines(imgray, [pts], True, (0,0,255))
cv.drawContours(img, snake2, -1, (0,0,255), 3)

plt.imshow(img_contours, 'gray')
plt.show()
cv.imwrite(img_contours, "images for contours features/active_contour.png")
#cv.imwrite("/Users/lianglianghao/Desktop/Failure ananlysis project/active_contour2.png", img)

#size of contours
area = cv.contourArea(snake2)
#Percentage of light area
img_threshold = cv.imread("images for contours features/threshold 127 full.png",0)
nonZeroPixel = cv.countNonZero(img_threshold)
light_area_percentage = nonZeroPixel / area #0.327133

#########contours of gray scale image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("images for contours features/threshold 127 full.png")


imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

s = np.linspace(0, 2*np.pi, 1000)
r = 550 + 550*np.sin(s)
c = 950 + 550*np.cos(s)
init = np.array([r, c]).T

#active contour
snake = active_contour(gaussian(imgray, 3), init, alpha=0.015, beta=10, gamma=0.001,
                       coordinates='rc')

#find all contours
contours, hierarchy = cv.findContours(imgray, 1, 2)

snake2 = snake.reshape((-1,1,2))
snake2 = np.array(snake2, np.int32)
tempSnake = np.copy(snake2[:,:,1])
snake2[:,:,1] = snake2[:,:,0]
snake2[:,:,0] = tempSnake
cv.drawContours(img, snake2, -1, (0,0,255), 3)
cv.drawContours(img, contours, -1, (0,0,255), 3)
#show the images
plt.imshow(img, 'gray')
plt.show()


##Canny edges
edged = cv.Canny(imgray, 10, 10)
plt.imshow(edged,'gray')
plt.show()
