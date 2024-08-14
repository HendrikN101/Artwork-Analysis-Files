# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 22:37:33 2021

@author: Hendrik
"""

import matplotlib.pyplot as plt
import numpy as np
#import analysisFunctionsGallery as af   # Required Python file.
import analysisFunctionsGalleryTesting as af   # Testing-branch Python file.
import tifffile as tf   # External download
import skimage.exposure as exposure
import time
from datetime import datetime
from alive_progress import alive_bar   # External download

plt.close('all')
plt.rc('figure', max_open_warning = 0)   # Removes 20 plot limit warning  
now = datetime.now()
initial_time = now.strftime('%H:%M:%S')
print('Initialisation Time:', initial_time)
print('')
st = time.time()  



# === HOW TO USE SCRIPT === #
#   1) Use ImageJ to sort B-scans into one C-scan (File > Import > Image Sequence...)
#   2) Import .tif (C-scan) from file location.
#   3) Run code.

# NOTE: Variables that can be adjusted...
#       Intensity Thresholds:   threshStrict, threshSmooth: Depends if scans are normalised or not: ~10k for unnormalised, ~200 for normalised.
#       Detection Distances:   m >= x, m<=x (Pixel-Height Analysis [Technique 1]); distance >= x (Dual-Line Analysis [Technique 2])



# IMPORT .TIF FILE:
image = tf.imread(r'D:\2023\Artwork (New)\1310 nm System\G5_Int_(2)0.tif')
im = exposure.rescale_intensity(image, in_range = 'image', out_range = (0, 255)).astype(np.uint8)   # Normalises tif.
del image   # Clearing Memory.
print('Data Loaded.')

# B-SCAN RANGE: (CAUTION: Will create a plot for every B-scan in range)
start = 0   # 0 Min
finish = (im.shape[0])   # (im.shape[0]) Max, will scan over full range.

# Manually set threshold intensities
intensity_avg = 190 #np.std(im)*2   # Calculates average intensity value to set as threshold
thresh = intensity_avg
print('Threshold:', thresh)

# Setting additional variables.
detected = False
print('Processing...')



# === PROJECTION VIEW === #

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



# === PROJECTION MAPPING === #

tech1_detections = 0    # Sets the number of detections for both techniques to start at zero.
tech2_detections = 0    # ""
distanceList = []   # Empty list to append distances between high- and low-order polynomial lines (Dual-Line)
stand_dev_list = []

plt.figure()
with alive_bar(finish) as bar:
    for i in range(start, finish):
        surf1 = af.surfaceDetect(im[i,0:im.shape[1],0:im.shape[2]],thresh=thresh)   # Obtains values for accurate and detailed surface line.
        surf2 = af.surfaceDetect2(im[i,0:im.shape[1],0:im.shape[2]],thresh=thresh)   # Obtains values for accurate, but 'smooth' surface line (Used for Dual-Line Method).
        d_surf=[(surf1[u+1]-surf1[u]) for u in range(len(surf1)-1)]   # Localised change along detailed surface line.
        plt.imshow(projection, cmap='gray')
        plt.axis('off')
        
        # Implement standard deviation for better surface quality measurement
        stand_dev = np.std(surf1)   # Calculates the standard deviation of the surface line.
        stand_dev_list.append(stand_dev)   # Creates list with the standard deviation from each B-scan.
    
    
    # SURFACE QUALITY (Run across whole C-scan for best results. Make sure no artifacts are interfering w/ surface lines)
    
        for b in range(0,im.shape[2]):
            distance = abs(surf1[b]-surf2[b])   # Measures the distances between the lines for each A-scan.
            distanceList.append(distance)   # Appends the measurement to a list for later use.
        
    
    # PIXEL-HEIGHT ANALYSIS (TECHNIQUE 1)
    
        d_surf1=[(surf1[i+1]-surf1[i]) for i in range(len(surf1)-1)]   # Localised change along surface line.
    
        for m in d_surf1:
            if m >= 2 in d_surf1[20:im.shape[2]-20]:   # Gradient threshold (m >= ..., -2, -1, 0, 1, 2, ...)
                detected = True
                max_value = max(d_surf1)
                max_index = d_surf1.index(max_value)
                if max_index in np.arange(20,im.shape[2]-20):   # Used to eliminate noise detection on left- and right-edges on B-scan.
                    plt.plot(max_index, i, marker=',', color='lightgreen')   # Plots green marker on en-face.
                    tech1_detections = tech1_detections + 1   # Increases detection count by 1.
                    break
            elif m <= -2 in d_surf1[20:im.shape[2]-20]:   # "" (Same as above, but for a decreasing defect).
                detected = True
                max_value = max(d_surf1)
                max_index = d_surf1.index(max_value)
                if max_index in np.arange(20,im.shape[2]-20):
                    plt.plot(max_index, i, marker=',', color='lightgreen')
                    tech1_detections = tech1_detections + 1
                    break
    
    
    # DUAL-LINE ANALYSIS (TECHNIQUE 2)
        
        for b in range(20,im.shape[2]-20):
            distance = abs(surf1[b]-surf2[b])   # Distance between two surface lines.
            if distance >= 5:   # If distance between surface lines exceeds a certain value.
                detected = True
                plt.imshow(projection, cmap='gray')
                plt.plot(b, i, marker=',', color='cyan')   # Plots blue marker on en-face.
                tech2_detections = tech2_detections + 1   # Increases detection count by 1.
                break
        bar()


plt.imshow(projection, cmap='gray')   # Displays the finalised en-face.
surfQual = sum(distanceList)/len(distanceList)   # Averages the distances by the length of the list.

surfqual = sum(stand_dev_list)/len(stand_dev_list)
del im   # Clearing Memory
del distanceList
del projection



# KEEP ACTIVE
print('')
print('Done!')
print('')
print('Pixel-Height Technique Detections: ', tech1_detections)   # Prints Pixel-Height detection count.
print('Dual-Line Technique Detections: ', tech2_detections)   # Prints Dual-Line detection count.
print('Surface Quality: ', round(1/surfQual, 3))   # Takes reciprocal of SQ-value to make a higher SQ-value be more positive.
print('Surface Quality (StD):', round(surfqual, 3))
if detected != True:
    print('Nothing of interest within the scanning range:', start, '-', finish)   # Prints if nothing is found.
    
print('')
elapsed_time = time.time() - st
print('Execution Time:', time.strftime('%H:%M:%S', time.gmtime(elapsed_time)), '(H:M:S)')


