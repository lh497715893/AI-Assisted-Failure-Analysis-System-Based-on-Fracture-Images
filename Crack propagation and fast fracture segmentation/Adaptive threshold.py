# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import cv2 as cv

#image = cv.imread('images for adaptive threshold/2110817386.jpg', 0)
#path = "images for adaptive threshold/aluminum_fatigue_fracture.jpg"
#path = "images for adaptive threshold/5AL 500℃ Nf65555 grayscale.jpg"
#path = "images for adaptive threshold/11AL 195 MPa Nf2176589 grayscale.jpg"
#path = "images for adaptive threshold/11AL 500℃ 235 MPa Nf grayscale.jpg"
#path = "images for adaptive threshold/5AL 500℃ Nf339419 grayscale.jpg"
#path = "images for adaptive threshold/origin_color.jpg"
#path = "images for adaptive threshold/5AL 140MPa Nf808162 grayscale.jpg"
path = "images for adaptive threshold/5AL 140MPa Nf808162 grayscale.jpg"

image = cv.imread(path, 0)
image = cv.medianBlur(image, 3)

#cv.imshow('image',image)
#cv.waitKey(0)
#cv.destroyAllWindows()

rows,cols = image.shape
roi = image[0:rows, 0:cols]
#result = image[0:rows, 0:cols]

ret, mask = cv.threshold(image, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
img2_fg = cv.bitwise_and(image,image,mask = mask)

ret, th1 = cv.threshold(image, 170, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY, 3, 2)
th3 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv.THRESH_BINARY,11,2)

#titles = ['original image', 'global threshing (v = 127)',
#            'Adaptive Meann Thresholding', 'Adaptive Gaussiann Thresholdinng']

titles = ['original image', 'global threshing (127)',
            'erosion', 'dilation', 'closing', 'gradient', 'img1_bg', 'img1_fg']

kernel = np.ones((4,4),np.uint8)
closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)

erosion = cv.erode(th1,kernel,iterations = 1)
dilation = cv.dilate(erosion,kernel,iterations = 1)
erosion = cv.erode(dilation,kernel,iterations = 1)
dilation = cv.dilate(erosion,kernel,iterations = 4)

images = [image, th1, erosion, dilation, closing, gradient, img1_bg, img2_fg]

for i in range(8):
    plt.subplot(4, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

#cv.imwrite(th1, "/Users/lianglianghao/Desktop/Failureexi ananlysis project/training data/failure type 2/1.jpg")
#######################################
#Load images for testing

def getShunduanImage(origin,threshold):
    #origin = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
    h = origin.shape[0]
    w = origin.shape[1]
    result = np.zeros([h,w],dtype=np.uint8)
    ret, th1 = cv.threshold(origin, threshold, 255, cv.THRESH_BINARY)
    kernel = np.ones((4,4),np.uint8)
    erosion = cv.erode(th1,kernel,iterations = 1)
    dilation = cv.dilate(erosion,kernel,iterations = 1)
    erosion = cv.erode(dilation,kernel,iterations = 1)
    dilation = cv.dilate(erosion,kernel,iterations = 4)
    for y in range(0, h):
        for x in range(0, w):
            if(dilation[y,x] != 0):
                result[y, x] = origin[y, x]
    return result

testing = getShunduanImage(image,210)
images = [testing, closing]

titles = ['dilation', 'closing']
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
            
cv.imwrite('/Users/lianglianghao/Desktop/Failure ananlysis project/Supplement images for segmentation/Testing for paper/5AL 140MPa Nf808162_kuozhan 2.jpg',testing)
def getKuozhanImage(ShunduanImage,threshold_start, threshold_end):
    h = ShunduanImage.shape[0]
    w = ShunduanImage.shape[1]
    result = np.zeros([h,w],dtype=np.uint8)
    #ret, th1 = cv.threshold(ShunduanImage, 0, threshold, cv.THRESH_BINARY)
    gray_filter = cv.inRange(ShunduanImage, threshold_start, threshold_end)
    kernel = np.ones((4,4),np.uint8)
    erosion = cv.erode(gray_filter,kernel,iterations = 1)
    dilation = cv.dilate(erosion,kernel,iterations = 2)
    for y in range(0, h):
        for x in range(0, w):
            if(dilation[y,x] != 0):
                result[y, x] = ShunduanImage[y, x]
    return result
            
