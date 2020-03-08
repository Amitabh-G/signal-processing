# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:58:03 2020

@author: amitabh.gunjan
"""
import cv2
import numpy as np
path = 'D:/other/Programming/python/signal-processing/machine-vision/topics/image-arithmetics/data/lena_color.png'
img = cv2.imread(path)
img
# See the image.
cv2.imshow('frame', img)
cv2.waitKey(0)

# Check the properties of image
# shape
print("The dimension of the image is:\n", img.shape)
# size = total number of pixels.
print("Number of pixels in the image is:\n ", img.size)
# image data type
print("Image data type is:\n", img.dtype)

# Accessing pixels
pix = img[100,100,]
p_blue = pix[0]
print("The blue pixel in pix:\n", p_blue)

# Modifying the pixel values:
img[100,100,] = [0,0,0]
img[100,100,]

# Splitting channels
b,g,r = cv2.split(img)

cv2.imshow('Blue ', b)
cv2.waitKey(0)

cv2.imshow('Green ', g)
cv2.waitKey(0)

cv2.imshow('Red ', r)
cv2.waitKey(0)

# Merging channels
img = cv2.merge((b,g,r))
cv2.imshow('Merged:',img)
cv2.waitKey(0)

# Another way
b = img[:,:,0]
cv2.imshow('Blue by subsetting:', b)
cv2.waitKey(0)