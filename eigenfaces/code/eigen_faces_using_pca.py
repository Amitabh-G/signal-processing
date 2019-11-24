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

nComp = 4
Xhat = np.dot(pca.transform(data_matrix)[:,:nComp], pca.components_[:nComp,:])
Xhat += mean_image
















