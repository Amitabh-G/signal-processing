# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:10:29 2019

@author: amitabh.gunjan
"""

import numpy as np
import sklearn.datasets, sklearn.decomposition
from PIL import Image
import glob

image_list = []
data_path = 'D:/other/Programming/python/signal-processing/eigenfaces/data/humans/faces94/faces94/male/9416994/*.jpg'
for filename in glob.glob(data_path):
    im=Image.open(filename)
    image_list.append(im)

image_list_resized = [im.resize((180,180), Image.ANTIALIAS) for im in image_list]
first_image = image_list_resized[0]

"""
convert to greyscale and flatten the images.
"""
first_greyscale = first_image.convert('L')
data_matrix = np.array([list(_image.convert('L').getdata()) for _image in image_list_resized]) # Does work like a charm. Use .getdata() for extracting pixel vals.

"""
Computing the mean image and removing the mean image from the list of images.
"""
mean_image = data_matrix.mean(0)
mean_image_reshape = mean_image.reshape(180, 180)
mean_doggo = Image.fromarray(np.uint8(mean_image_reshape), 'L')
mean_doggo.show()

pca = sklearn.decomposition.PCA()
pca.fit(data_matrix)

nComp = 2
Xhat = np.dot(pca.transform(data_matrix)[:,:nComp], pca.components_[:nComp,:])
Xhat += mean_image


def normalize_images(x):
    x = ((255/(x.max() - x.min()))*(x - x.max())) + 255
    return x


normalized_difference_doggos = np.apply_along_axis(normalize_images, axis=0, arr=Xhat)

list_diff_images = [Image.fromarray(np.uint8(x.reshape(180, 180)), 'L') for x in normalized_difference_doggos]
#for i in list_diff_images:
#    i.show()


"""
Do the same for an image.
"""
input_image_path = 'D:/other/Programming/python/signal-processing/eigenfaces/data/humans/faces94/faces94/male/9416994/9416994.11.jpg'
im=Image.open(input_image_path)
im = im.resize((180,180), Image.ANTIALIAS)
    
im.convert('L').show()
image_vector = np.array(list(im.convert('L').getdata()))
#image_vector_diff = image_vector - mean_image
image_vector_diff = image_vector.reshape(1, 32400)


nComp = 5
img_transform = pca.transform(image_vector_diff)
print(img_transform[0][:nComp])


Xhat = np.dot(pca.transform(image_vector_diff)[0][:nComp], pca.components_[:nComp,:])


Xhat += mean_image # add the mean image to the reconstructed image.

"""
Getting the exact reconstruction using pca library implementation.
"""

reconstructed_image_reshaped = Xhat.reshape(180, 180)
rescaled_image = ((255/(reconstructed_image_reshaped.max() - reconstructed_image_reshaped.min()))*(reconstructed_image_reshaped - reconstructed_image_reshaped.max())) + 255
reconstructed_doggo = Image.fromarray(np.uint8(rescaled_image), 'L')
reconstructed_doggo.show()


