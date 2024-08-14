# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 22:37:33 2021

@author: Hendrik
"""

import matplotlib.pyplot as plt
import numpy as np
import analysisFunctionsGallery as af
import scipy
import tifffile as tf   # External download

plt.close('all')
plt.rc('figure', max_open_warning = 0)   # Removes 20 plot limit warning   



# === HOW TO USE SCRIPT === #
#   1) Use ImageJ to sort B-scans into one C-scan (File > Import > Image Sequence...)
#   2) Import .tif (C-scan) from file location.
#   3) Select B-scan range for analysis.
#   4) Run code.

# NOTE: Variables that may need adjusting...
#       threshStrict, threshSmooth: 1310nm intensity values much smaller than 850nm, requires different max intensity percentages.
#       distance >= x, distance <= x, (Dual-Line Analysis): 1310nm resolution much smaller than 850nm, meaning smaller bumps need to be detected.



# IMPORT .TIF FILE:
im = tf.imread(r'C:\Users\User\Desktop\Honours Project\840 nm System (Lumedica)\Third Artwork\A3-2\Reslice of ThirdPiece_Front2 (Crop) R=15.tif')

# B-SCAN RANGE: (CAUTION: Will create a plot for every B-scan in range)
start = 0   # 0 Min (50 for Lumedica due to aliasing)
finish = (im.shape[0]-1)   # (im.shape[0]-1) Max

# Manually set threshold intensities
threshStrict = 10000 #10000   # ~10000 for standard scans, ~250 for normalised scans, ~20000 for Lab 850 scans
threshSmooth = 10000 #10000   # ""

# # Automatically determines threshold intensity. (Currently does not work)
# maxInt = np.amax(im)
# print(maxInt)
# threshStrict = maxInt/2.5  # 2.5 for 850nm, 1.2 for 1310nm
# threshSmooth = maxInt/3  # 6 for 850nm, 1.5 for 1310nm

# Setting additional variables.
detected = False
x = np.arange(0,im.shape[2],1)
print('Processing...')





# === PROJECTION VIEW === #

stackedImages = np.array(())
stackedImages = np.append(stackedImages, im)

finalImage = stackedImages.reshape(int(np.sum(im.shape[0])),int(im.shape[1]),np.size(im,2))

projection = np.sum(finalImage[:,:,:],1)
plt.figure()
#plt.title('Projection View')
plt.axis('off')
plt.imshow(projection, cmap='gray')
plt.show()





# # === SURFACE & DETECTION MAPPING === #

# for n in range(start, finish + 1):  # +1 to include the end of the range in the list
#     x = x - start + 1;  # make x start at 1
#     surf1 = af.surfaceDetect(im[n,0:im.shape[1],0:im.shape[2]],thresh=threshStrict,buffer=10,skip=0)
#     surf2 = af.surfaceDetect2(im[n,0:im.shape[1],0:im.shape[2]],thresh=threshSmooth,buffer=10,skip=0)
#     plt.figure()
#     #plt.ylim(400)
#     plt.imshow(im[n,:,:],cmap='gray')
#     x = np.arange(0,im.shape[2],1)   # Fixes x-axis plotting issue
#     line1 = plt.plot(x,surf1,'-', color='red', linewidth=1)   # [x,im.shape[2]] for full width
#     plt.ylim(0, im.shape[1])
#     line2 = plt.plot(x,surf2,'-', color='cyan', linewidth=1)   # [x,im.shape[2]] for full width
#     plt.ylim(im.shape[1], 0)
#     plt.title('B-Scan: n, '+ str(n))
            
    

# # PIXEL-HEIGHT ANALYSIS (TECHNIQUE 1)
#     d_surf1=[(surf1[i+1]-surf1[i]) for i in range(len(surf1)-1)]   # localised change
      
#     for m in d_surf1:
#         if m >= 4 in d_surf1[20:im.shape[2]-20]:   # m = gradient threshold (..., -1, -2, 0, 1, 2, ...), im.shape[2]-20 removes 20px of noise from end of image
#             max_value = max(d_surf1)
#             max_index = d_surf1.index(max_value)
#             if max_index in np.arange(20,im.shape[2]-20):   # ignores start and end noise
#                 plt.plot(x[max_index], surf1[max_index], marker='+', color='lightgreen')
#                 print('[Pixel-Height] Possible point of interest detected!   B-scan #', n, '   Location:', 'x =', x[max_index], 'y =', surf1[max_index])
#                 detected = True
#                 break
#         elif m <= -4 in d_surf1[20:im.shape[2]-20]:   # -4 default
#             max_value = max(d_surf1)
#             max_index = d_surf1.index(max_value)
#             if max_index in np.arange(20,im.shape[2]-20):
#                 plt.plot(x[max_index], surf1[max_index], marker='+', color='lightgreen')
#                 print('[Pixel-Height] Possible point of interest detected!   B-scan #', n, '   Location:', 'x =', x[max_index], 'y =', surf1[max_index])
#                 detected = True
#                 break



# # DUAL-LINE ANALYSIS (TECHNIQUE 2)    
#     for b in range(20,im.shape[2]-20): # 0,im.shape[2] for full width
#         distance = abs(surf1[b]-surf2[b])
#         if distance >= 16: # 16 default
#             Detected = True
#             #print(distance, x[surf1[b]], surf1[b])
#             plt.plot(x[b], surf1[b], marker='+', color='cyan')
#             print('[Dual-Line] Possible point of interest detected!   B-scan #', n, '   Location:', 'x =', x[b], 'y =', surf1[b])
#             break





# === PROJECTION MAPPING === #

tech1_detections = 0    # Sets the number of detections for both techniques to start at zero.
tech2_detections = 0    # ""
distanceList = []   # Empty list to append distances between high- and low-order polynomial lines (Dual-Line)

plt.figure()
for i in range(start, finish + 1):
    x = x - start + 1;  # make x start at 1
    surf1 = af.surfaceDetect(im[i,0:im.shape[1],0:im.shape[2]],thresh=threshStrict,buffer=10,skip=5)
    surf2 = af.surfaceDetect2(im[i,0:im.shape[1],0:im.shape[2]],thresh=threshSmooth,buffer=10,skip=5)
    d_surf=[(surf1[u+1]-surf1[u]) for u in range(len(surf1)-1)]
    plt.imshow(projection, cmap='gray')
    #plt.title('Projection with Irregularities Mapped')
    plt.axis('off')


# SURFACE QUALITY (Run across whole C-scan for best results. Make sure no artifacts are interfering w/ surface lines)

    for b in range(0,im.shape[2]):
        distance = abs(surf1[b]-surf2[b])
        distanceList.append(distance)
    


# PIXEL-HEIGHT ANALYSIS (TECHNIQUE 1)

    d_surf1=[(surf1[i+1]-surf1[i]) for i in range(len(surf1)-1)]   # localised change

    for m in d_surf1:
        if m >= 4 in d_surf1[20:im.shape[2]-20]:   # gradient threshold (..., -1, -2, 0, 1, 2, ...)
            detected = True
            max_value = max(d_surf1)
            max_index = d_surf1.index(max_value)
            if max_index in np.arange(20,im.shape[2]-20):   # ignores start and end noise
                plt.plot(max_index, i, marker=',', color='lightgreen')
                tech1_detections = tech1_detections + 1
                break
        elif m <= -4 in d_surf1[20:im.shape[2]-20]: # default 4
            detected = True
            max_value = max(d_surf1)
            max_index = d_surf1.index(max_value)
            if max_index in np.arange(20,im.shape[2]-20):
                plt.plot(max_index, i, marker=',', color='lightgreen')
                tech1_detections = tech1_detections + 1
                break



# DUAL-LINE ANALYSIS (TECHNIQUE 2)
    
    for b in range(40,im.shape[2]-40): # 0,im.shape[2]
        distance = abs(surf1[b]-surf2[b])
        if distance >= 16: #16 default, 6 for Lab 850 scans
            detected = True
            #print(distance, x[surf1[b]], surf1[b])
            plt.imshow(projection, cmap='gray')
            plt.plot(b, i, marker=',', color='cyan')
            tech2_detections = tech2_detections + 1
            break



plt.imshow(projection, cmap='gray')
surfQual = sum(distanceList)/len(distanceList)   # Averages the distances by the length of the list.



# KEEP ACTIVE
print('')
print('Done!')
print('')
print('Pixel-Height Technique Detections: ', tech1_detections)
print('Dual-Line Technique Detections: ', tech2_detections)
print('Surface Quality: ', round(1/surfQual, 3))
if detected != True:
    print('Nothing of interest within the scanning range:', start, '-', finish)