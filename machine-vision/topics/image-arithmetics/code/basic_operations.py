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

# Region of image - take the near eye region and superpose it somewhere.
near_eye = img[240:340, 240:340]
img[100:200, 100:200] = near_eye
cv2.imshow('region of image:', img)
cv2.waitKey(0)

# Padding images.To create a border around the images.
# Used in convolution operations.
from matplotlib import pyplot as plt
color = [205,0,0]
img1 = cv2.imread(path)

"""
cv2.BORDER_CONSTANT - Adds a constant colored border. The value should be given as next argument.
cv2.BORDER_REFLECT - Border will be mirror reflection of the border elements, like this : fedcba|abcdefgh|hgfedcb
cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT - Same as above, but with a slight change, like this : gfedcb|abcdefgh|gfedcba
cv2.BORDER_REPLICATE - Last element is replicated throughout, like this: aaaaaa|abcdefgh|hhhhhhh
cv2.BORDER_WRAP - Can’t explain, it will look like this : cdefgh|abcdefgh|abcdefg

"""

replicate = cv2.copyMakeBorder(img1,30,30,30,30,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,30,30,30,30,cv2.BORDER_CONSTANT,value=color)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()













