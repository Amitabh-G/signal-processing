# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:44:45 2019

@author: amitabh.gunjan
"""

"""
Creating eigenfaces for dogs' images.
    create one data matrix and repesent an image as a row vector  
    in the whola data matrix.
"""
# load the images.

from PIL import Image
import glob

image_list = []
data_path = 'D:/other/Programming/python/signal-processing/eigenfaces/data/*.jpg'
for filename in glob.glob(data_path):
    im=Image.open(filename)
    image_list.append(im)

# resize the images.
image_list_resized = [im.resize((200,200), Image.ANTIALIAS) for im in image_list]
import numpy as np
first_image = image_list_resized[0]
first_image.show()

# Get the pixel data from the image.
first_image_px_list = list(first_image.getdata())
first_image_px_list
first_image_px = first_image.load()

# Access the pixel values by positions.
first_image_px[0,0]
first_image_px_list[0]

# convert to greyscale and flatten the images.
first_greyscale = first_image.convert('L')
first_greyscale.show() # look at the doggo.

#data_matrix = np.array([np.array(_image.convert('L').load()) for _image in image_list_resized]) # does not work. 
data_matrix = np.array([list(_image.convert('L').getdata()) for _image in image_list_resized]) # Does work like a charm. Use .getdata() for extracting pixel vals.

img_to_row_matrix = data_matrix.flatten().reshape(10, 40000)

cov_matrix = np.cov(img_to_row_matrix)
from numpy import linalg as LA
evals, eigen_dogs_vectors = LA.eig(cov_matrix)
evals
eigen_dogs_vectors
# Get the eigenvectors of A^T*A.

# Difference faces -- images - mean of them.
img_to_row_matrix_centered = img_to_row_matrix - img_to_row_matrix.mean(axis=1, keepdims=True)
img_to_row_matrix_centered_transposed = np.transpose(img_to_row_matrix_centered)

# Both the transformations work fine. Need to inspect for any differences.

eigen_dogs_vectors_transformed = [np.transpose(img_to_row_matrix).dot(e_doggo) for e_doggo in eigen_dogs_vectors]
eigen_dogs_vectors_transformed = [img_to_row_matrix_centered_transposed.dot(e_doggo) for e_doggo in eigen_dogs_vectors]

eigen_dogs_vec_reshaped = [e_doggo.reshape(200, 200) for e_doggo in eigen_dogs_vectors_transformed]


# Scale the eigenvectors back to the original scale to reproduce the images
#newvalue= (max'-min')/(max-min)*(value-max)+max'

rescaled_evectors = [((255/(_dogs.max() -_dogs.min()))*(_dogs - _dogs.max())) + 255 for _dogs in eigen_dogs_vec_reshaped]
eigen_dogs_images = [Image.fromarray(np.uint8(_dogs), 'L') for _dogs in rescaled_evectors]

# Show the images.
for i in eigen_dogs_images:
    i.show()









