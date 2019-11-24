# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:00:48 2019

@author: amitabh.gunjan
"""

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
from PIL import Image
import glob

image_list = []
data_path = 'D:/other/Programming/python/signal-processing/eigenfaces/data/humans/faces94/faces94/male/9416994/*.jpg'
for filename in glob.glob(data_path):
    im=Image.open(filename)
    image_list.append(im)

image_list_resized = [im.resize((180,180), Image.ANTIALIAS) for im in image_list]
import numpy as np
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

def remove_mean_image( x ):
    return x - (sum(x)/len(x))

difference_doggos = np.apply_along_axis(remove_mean_image, axis=0, arr=data_matrix )

def normalize_images(x):
    x = ((255/(x.max() - x.min()))*(x - x.max())) + 255
    return x


normalized_difference_doggos = np.apply_along_axis(normalize_images, axis=0, arr=difference_doggos)

list_diff_images = [Image.fromarray(np.uint8(x.reshape(180, 180)), 'L') for x in normalized_difference_doggos]

#for i in list_diff_images:
#    i.show()
    
#cov_matrix = np.cov(difference_doggos)
cov_matrix = np.matmul(difference_doggos, difference_doggos.T)
from numpy import linalg as LA
evals, eigenvec = LA.eig(cov_matrix)
eigen_dogs_vectors = np.negative(eigenvec)



"""
Get the eigenvectors of A^T*A and reshape the evectors to create the ghost images of the doggos.
"""
eigen_dogs_vectors_transformed = [np.transpose(difference_doggos).dot(e_doggo) for e_doggo in eigen_dogs_vectors]
eigen_dogs_vec_reshaped = [e_doggo.reshape(180, 180) for e_doggo in eigen_dogs_vectors_transformed]

"""
Scale the eigenvectors back to the original scale to reproduce the images using the belwo formula
    newvalue= (max'-min')/(max-min)*(value-max)+max'
"""
rescaled_evectors = [((255/(_dogs.max() - _dogs.min()))*(_dogs - _dogs.max())) + 255 for _dogs in eigen_dogs_vec_reshaped]
dogs_face_space = [Image.fromarray(np.uint8(_dogs), 'L') for _dogs in rescaled_evectors]

#for i in dogs_face_space:
#    i.show()
def project_image_to_face_space(input_image_path, face_space):
    im=Image.open(input_image_path)
    im = im.resize((180,180), Image.ANTIALIAS)
    
    im.convert('L').show()
    image_vector = np.array(list(im.convert('L').getdata()))
    """
    Subtract mean image from the doggo.
    """
    image_vector = image_vector - mean_image
    image_vector = image_vector.reshape(1, 32400)
    
    coefficients = np.dot(image_vector, face_space)
    '''
    Instead of dot product just add the vectors after multiplying the respective coefficients. As provided on this website.
        https://jeremykun.com/2011/07/27/eigenfaces/
        
    '''
    reconstructed_image = np.dot(image_vector, face_space).dot(face_space.T)

    """
    Add mean doggo to the reconstructed doggo.
    """
#    reconstructed_image = np.add(reconstructed_image, mean_image.reshape(1, 32400))
    reconstructed_image_reshaped = reconstructed_image.reshape(180, 180)
    rescaled_image = ((255/(reconstructed_image_reshaped.max() - reconstructed_image_reshaped.min()))*(reconstructed_image_reshaped - reconstructed_image_reshaped.max())) + 255
    reconstructed_doggo = Image.fromarray(np.uint8(rescaled_image), 'L')
    reconstructed_doggo.show()
    return reconstructed_doggo

input_image_path = 'D:/other/Programming/python/signal-processing/eigenfaces/data/humans/faces94/faces94/male/9416994/9416994.11.jpg'
eigen_dogs_vectors_transformed_matrix = np.matrix(eigen_dogs_vectors_transformed)
eigen_dogs_vectors_transformed_matrix_t = eigen_dogs_vectors_transformed_matrix.T
reconstructed_doggo = project_image_to_face_space(input_image_path=input_image_path, face_space=eigen_dogs_vectors_transformed_matrix_t)


reconstructed_doggos = np.dot(np.dot(normalized_difference_doggos, eigen_dogs_vectors_transformed_matrix_t), eigen_dogs_vectors_transformed_matrix)


"""
Get the images reconstructed back.
"""
reconstructed_doggos_rescaled = [((255/(_dogs.max() - _dogs.min()))*(_dogs - _dogs.max())) + 255 for _dogs in reconstructed_doggos]
reconstructed_images_reshaped = [e_doggo.reshape(180, 180) for e_doggo in reconstructed_doggos_rescaled]
reconstructed_images = [Image.fromarray(np.uint8(_dogs), 'L') for _dogs in reconstructed_images_reshaped]
#for i in reconstructed_images:
#    i.show() # Images are not exactly the same but quite alike.