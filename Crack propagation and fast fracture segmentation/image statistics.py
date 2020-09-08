import numpy as np
import cv2 as cv

img = cv.imread("/Users/lianglianghao/Desktop/Failure ananlysis project/origin.jpg", 0)
img_threshold = cv.imread("/Users/lianglianghao/Desktop/Failure ananlysis project/threshold 127 full.png",0)

#calculate white pixels
nonZeroPixel = cv.countNonZero(img_threshold)

nonZeroImg = cv.countNonZero(img)

y = img_threshold.shape[0]
x = img_threshold.shape[1]

nonZeroPercentage = nonZeroPixel / (x*y)

kuozhanqubili = nonZeroPixel/nonZeroImg

shunduanqubili = (nonZeroImg - nonZeroPixel)/nonZeroImg

bili = kuozhanqubili / shunduanqubili
