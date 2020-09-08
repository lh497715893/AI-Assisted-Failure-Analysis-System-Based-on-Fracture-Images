# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv




#Read template images for shunduan and kuozhan
pathOfTemplate = 'images for image statistics/5AL 140MPa Nf808162 grayscale.jpg'
pathOfShunduanTem = 'images for image statistics/5AL 140MPa Nf808162 grayscale_shunduan_template.jpg'
pathOfKuoZhanTem = 'images for image statistics/5AL 140MPa Nf808162 grayscale_kuozhan_template.jpg'

template1 = cv.imread(pathOfTemplate, 0)
shunduan_temp1 = cv.imread(pathOfShunduanTem, 0)
kuozhan_temp1 = cv.imread(pathOfKuoZhanTem, 0)

#calculate height and width of image
h, w = template1.shape

#Read test images for shunduan and kuozhan
pathOfShunduan = 'images for image statistics/5AL 140MPa Nf808162_shunduan 2.jpg'
pathOfKuoZhan = 'images for image statistics/5AL 140MPa Nf808162_kuozhan 2.jpg'

shunduan_test1 = cv.imread(pathOfShunduan, 0)
kuozhan_test1 = cv.imread(pathOfKuoZhan, 0)

#calculate pixels of template image
#total_pixel = np.count_nonzero(template1[:]>=5)
#template1_pixel = np.count_nonzero(template1[:]>5)
shunduan_temp1_pixel = np.count_nonzero(shunduan_temp1[:]>5)
kuozhan_temp1_pixel = np.count_nonzero(kuozhan_temp1[:]>5)
total_pixel = shunduan_temp1_pixel + kuozhan_temp1_pixel 

#calculate pixels of shunduan and kuozhan image
shunduan_test1_pixel = np.count_nonzero(shunduan_test1[:]>5)
kuozhan_test1_pixel = np.count_nonzero(kuozhan_test1[:]>5)

#calculate the overlap between template and shunduan，kuozhan image
overlap_shunduan = 0
for i in range(h):
    for j in range(w):
        if( (shunduan_temp1[i][j] > 5) & (shunduan_test1[i][j] > 5) ):
            overlap_shunduan = overlap_shunduan + 1

overlap_kuozhan = 0
for i in range(h):
    for j in range(w):
        if( (kuozhan_temp1[i][j] > 5) & (kuozhan_test1[i][j] > 5) ):
            overlap_kuozhan = overlap_kuozhan + 1

#calculate TT(right overlap)
TT_shunduan = overlap_shunduan / total_pixel
TT_kuozhan = overlap_kuozhan / total_pixel

#calculate TF(predict truth,but actually fault)

wrong_overlap_shunduan = shunduan_test1_pixel - overlap_shunduan
wrong_overlap_kuozhan = kuozhan_test1_pixel - overlap_kuozhan

TF_shunduan = wrong_overlap_shunduan / total_pixel
TF_kuozhan = wrong_overlap_kuozhan / total_pixel

#calculate FT(predict fault,but actually truth)

not_overlap_shunduan = shunduan_temp1_pixel - overlap_shunduan
not_overlap_kuozhan = kuozhan_temp1_pixel - overlap_kuozhan

FT_shunduan = not_overlap_shunduan / total_pixel
FT_kuozhan = not_overlap_kuozhan / total_pixel

#calculate TT(predict fault and actually fault)

right_not_shunduan = total_pixel - overlap_shunduan - wrong_overlap_shunduan - not_overlap_shunduan
right_not_kuozhan = total_pixel - overlap_kuozhan - wrong_overlap_kuozhan - not_overlap_kuozhan

FF_shunduan = right_not_shunduan / total_pixel
FF_kuozhan = right_not_kuozhan / total_pixel

#print:
print("TT_shunduan:"+str(TT_shunduan))
print("TF_shunduan:"+str(TF_shunduan))
print("FT_shunduan:"+str(FT_shunduan))
print("FF_shunduan:"+str(FF_shunduan))

print("TT_kuozhan_:"+str(TT_kuozhan))
print("TF_kuozhan:"+str(TF_kuozhan))
print("FT_kuozhan:"+str(FT_kuozhan))
print("FF_kuozhan:"+str(FF_kuozhan))
print("---------------------------")
#print:
print("overlap_shunduan:"+str(overlap_shunduan))
print("wrong_overlap_shunduan:"+str(wrong_overlap_shunduan))
print("not_overlap_shunduan:"+str(not_overlap_shunduan))
print("right_not_shunduan:"+str(right_not_shunduan))

print("overlap_kuozhan:"+str(overlap_kuozhan))
print("wrong_overlap_kuozhan:"+str(wrong_overlap_kuozhan))
print("not_overlap_kuozhan:"+str(not_overlap_kuozhan))
print("right_not_kuozhan:"+str(right_not_kuozhan))



#for testing only
temp = np.zeros((h,w),dtype=np.uint8)
for i in range(h):
    for j in range(w):
        if(shunduan_temp1[i][j] > 5):
            temp[i][j] = 255

cv.imshow('temp', temp)
