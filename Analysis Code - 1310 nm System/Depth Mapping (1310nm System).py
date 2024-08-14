# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:27:26 2022

@author: Hendrik
"""

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf #install tiffle: pip install tiffile
import analysisFunctionsGallery as af
import cv2

# Range of B-scan values you want processed...
start = 0
finish = 219   # 219 MAX for 680x512 B-scans, 679 MAX for 220x512 B-scans

x = np.arange(0,680,1)

# FILES: F:/2021/Art Gallery/G4/output_files/G4_220_(2)0_00/Int_(2)0.tif,   E:/Mini-Project/Artwork/G3/output_files/G3_220_(2)0_00/Int_(2)0.tif
#        F:/2021/Art Gallery/G5/output_files/G5_220_(2)0_00/Int_(2)0.tif,   E:/Mini-Project/Artwork/G2/output_files/G2_240_(2)0_00/Int_(2)0.tif

# TOP
im = tf.imread('E:/Mini-Project/Artwork/G3/output_files/G3_220_(2)0_00/Int_(2)0.tif')   # Reads from .tif

test_proj_top = []   # Creates an empty list to append arrays into
for i in range(start, finish + 1):
     x = x - start + 1;
     #surf = af.surfaceDetect(im[i,0:680,0:512],thresh=100,padSize=0,scale=1,buffer=10,skip=5)   # For 220x512 B-scans
     surf = af.surfaceDetect(im[i,0:512,0:680],thresh=100,padSize=0,scale=1,buffer=10,skip=5)   # For 680x512 B-scans
     test_proj_top.append(surf)   # Appends each array into list
     
#plt.imshow(test_proj_top, cmap='gray')


# FILES: F:/2021/Art Gallery/G4/output_files/G4_220_02_00/Int_02.tif,   E:/Mini-Project/Artwork/G3/output_files/G3_220_02_00/Int_02.tif
#        F:/2021/Art Gallery/G5/output_files/G5_220_02_00/Int_02.tif,   E:/Mini-Project/Artwork/G2/output_files/G2_220_02_00/Int_02.tif

# BOTTOM
im = tf.imread('E:/Mini-Project/Artwork/G3/output_files/G3_220_02_00/Int_02.tif')   # Reads from .tif

test_proj_bottom = []   # Creates an empty list to append arrays into
for i in range(start, finish + 1):
     x = x - start + 1;
     #surf = af.surfaceDetect(im[i,0:680,0:512],thresh=100,padSize=0,scale=1,buffer=10,skip=5)   # For 220x512 B-scans
     surf = af.surfaceDetect(im[i,0:512,0:680],thresh=100,padSize=0,scale=1,buffer=10,skip=5)   # For 680x512 B-scans
     test_proj_bottom.append(surf)   # Appends each array into list
     
#plt.imshow(test_proj_top, cmap='gray')


projection_image = np.vstack((test_proj_top, test_proj_bottom))   # Stacks the two lists together (np.hstack for 220x512 B-scans)
# projection_image = cv2.flip(projection_image, 1)   # For G5 to flip
# projection_image = np.rot90(projection_image, 2)   # For G3 rotation
plt.imshow(projection_image, cmap='gray')