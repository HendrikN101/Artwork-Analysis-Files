# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:27:26 2022

@author: Hendrik
"""

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf #install tifffile: pip install tifffile
import analysisFunctionsGalleryTesting as af
import skimage.exposure as exposure
import time
from datetime import datetime
from tqdm import tqdm   # External download

now = datetime.now()
initial_time = now.strftime('%H:%M:%S')
print('Initialisation Time:', initial_time)
print('Processing...')
print('')
st = time.time()



# Importing .TIF file
image = tf.imread(r'D:\2023 Scans\TIFS\Other Artworks\RibofAdam1.tif')   # Reads from .tif
im = exposure.rescale_intensity(image, in_range = 'image', out_range = (0, 255)).astype(np.uint8)   # Normalises tif into an 8-bit format.
del image   # Clearing Memory.

# Range of B-scans to be processed
start = 0
finish = im.shape[0]

# Setting threshold intensity
# intensity_avg = 40 #np.std(im)*2   # Uses the standard deviation of the images intensity values to use as automatic threshold.
# thresh = np.floor(intensity_avg)
# print('Threshold:', round(thresh, 0))


' EN-FACE '

stackedImages = np.array(())   # Creates empty array.
stackedImages = np.append(stackedImages, im)   # Appends the image data into the array.
finalImage = stackedImages.reshape(int(np.sum(im.shape[0])),int(im.shape[1]),np.size(im,2))   # Reshapes the data in array.
projection = np.sum(finalImage[:,:,:],1)   # Forms an en-face image of the C-scan image data.
plt.figure()
plt.axis('off')
plt.imshow(projection, cmap='gray')
plt.show()
del finalImage   # Clearing Memory.
del stackedImages


' SURFACE ELEVATION MAPPING'

test_proj_top = []   # Creates an empty list to append arrays into
for i in tqdm(range(start, finish)):
    thresh = np.std(im[i,:,:])   # Variable threshold using standard deviation for each B-scan.
    surf = af.surfaceDetect(im[i,0:im.shape[1],0:im.shape[2]],thresh=thresh)
    test_proj_top.append(surf)   # Appends each array into list


' PLOTTING '

plt.imshow(test_proj_top, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.title('Surface Elevation Map')
plt.show()

elapsed_time = time.time() - st
print('Done!')
print('Execution Time:', time.strftime('%H:%M:%S', time.gmtime(elapsed_time)), '(H:M:S)')
del im   # Clearing Memory