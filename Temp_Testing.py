# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:23:57 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
#import analysisFunctionsGallery as af   # Required Python file.
#import analysisFunctionsGalleryTesting as af   # Testing-branch Python file.
import tifffile as tf   # External download
import skimage.exposure as exposure
from scipy import signal,ndimage
import scipy as sci
from alive_progress import alive_bar
import time

for x in 1000, 1500, 700, 0:
   with alive_bar(x) as bar:
       for i in range(1000):
           time.sleep(.005)
           bar()

# # IMPORT .TIF FILE:
# im = tf.imread(r'D:\2023\Artwork (New)\840 nm System (Lumedica)\Virgin and the Child\Reslice of ThirdPiece_Front2 (Crop) R=15.tif')
