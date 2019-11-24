# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:08:20 2019

@author: amitabh.gunjan
"""

# Images.

from PIL import Image
import numpy as np
img = Image.open( 'D:/other/others/pics/profilePic.PNG' )
img.load()
img.size
img.mode

data = np.asarray( img, dtype="int32" )
data.shape
