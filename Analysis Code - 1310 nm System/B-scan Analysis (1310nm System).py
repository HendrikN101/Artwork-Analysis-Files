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
finish = 5   # (im.shape[0]) Max - will scan over full range (Beware! Will print every B-scan).

# Manually set threshold intensities
intensity_avg = 175 #np.std(im)*2   # Calculates average intensity value to set as threshold
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



# === SURFACE & DETECTION MAPPING === #

with alive_bar(finish) as bar:
    for n in range(start, finish + 1):  # Loop prints defect analysed B-scans from selected range.
        #x = x - start + 1;  # make x start at 1
        surf1 = af.surfaceDetect(im[n,0:im.shape[1],0:im.shape[2]],thresh=thresh)   # Obtains values for accurate and detailed surface line.
        surf2 = af.surfaceDetect2(im[n,0:im.shape[1],0:im.shape[2]],thresh=thresh)   # Obtains values for accurate, but 'smooth' surface line (Used for Dual-Line Method).
        plt.figure()
        #plt.ylim(400)
        plt.imshow(im[n,:,:],cmap='gray')
        x = np.arange(0,im.shape[2],1)   # Sets x-axis to horizontal length of B-scan.
        line1 = plt.plot(x,surf1,'-', color='red', linewidth=1)   # Plots detailed surface line.
        plt.ylim(0, im.shape[1])
        line2 = plt.plot(x,surf2,'-', color='cyan', linewidth=1)   # Plots 'smooth' surface line.
        plt.ylim(im.shape[1], 0)
        plt.title('B-Scan: n, '+ str(n))
                
        
    # PIXEL-HEIGHT ANALYSIS (TECHNIQUE 1)
        d_surf1=[(surf1[i+1]-surf1[i]) for i in range(len(surf1)-1)]   # localised change
          
        for m in d_surf1:
            if m >= 2 in d_surf1[20:im.shape[2]-20]:   # Gradient threshold (m >= ..., -2, -1, 0, 1, 2, ...)
                max_value = max(d_surf1)
                max_index = d_surf1.index(max_value)
                if max_index in np.arange(20,im.shape[2]-20):   # Used to eliminate noise detection on left- and right-edges on B-scan.
                    plt.plot(x[max_index], surf1[max_index], marker='+', color='lightgreen')   # Plots green marker on en-face.
                    print('[Pixel-Height] Possible point of interest detected!   B-scan #', n, '   Location:', 'x =', x[max_index], 'y =', surf1[max_index])
                    detected = True
                    break
            elif m <= -2 in d_surf1[20:im.shape[2]-20]:   # "" (Same as above, but for a decreasing defect).
                max_value = max(d_surf1)
                max_index = d_surf1.index(max_value)
                if max_index in np.arange(20,im.shape[2]-20):
                    plt.plot(x[max_index], surf1[max_index], marker='+', color='lightgreen')
                    print('[Pixel-Height] Possible point of interest detected!   B-scan #', n, '   Location:', 'x =', x[max_index], 'y =', surf1[max_index])
                    detected = True
                    break
    
    
    # DUAL-LINE ANALYSIS (TECHNIQUE 2)    
        for b in range(20,im.shape[2]-20):
            distance = abs(surf1[b]-surf2[b])   # Distance between two surface lines.
            if distance >= 5:   # If distance between surface lines exceeds a certain value.
                Detected = True
                #print(distance, x[surf1[b]], surf1[b])
                plt.plot(x[b], surf1[b], marker='+', color='cyan')   # Plots blue marker on en-face.
                print('[Dual-Line] Possible point of interest detected!   B-scan #', n, '   Location:', 'x =', x[b], 'y =', surf1[b])
                break
        bar()



# KEEP ACTIVE
print('')
print('Done!')
print('')
if detected != True:
    print('Nothing of interest within the scanning range:', start, '-', finish)   # Prints if nothing is found.
    
print('')
elapsed_time = time.time() - st
print('Execution Time:', time.strftime('%H:%M:%S', time.gmtime(elapsed_time)), '(H:M:S)')
del im   # Clearing Memory