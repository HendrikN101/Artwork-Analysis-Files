# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:46:16 2023

@author: Hendrik
"""

import cv2
import numpy as np
import skimage.exposure as exposure
import tifffile as tf

# Read tiff
img = tf.imread(r'D:\2023\Artwork (New)\850 nm System\Front\area2_t1.tif')

#img_norm = 255*(img - img.min())/(img.max() - img.min())
img_norm = exposure.rescale_intensity(img, in_range='image', out_range=(0,255)).astype(np.uint8)
tf.imwrite('D:/2023/Artwork (New)/850 nm System/Front/Front/Normalised area2_t1.tif', img_norm)

# Normalize image to 8-bit range
#img_norm = exposure.rescale_intensity(img, in_range='image', out_range=(0,255)).astype(np.uint8)
#cv2.imwrite('Normalised area2_t1', img_norm)