# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:26:30 2023

@author: Hendrik
"""

import numpy as np
import matplotlib.pyplot as plt
#import analysisFunctionsGallery as af   # Required Python file.
#import analysisFunctionsGalleryTesting as af   # Testing-branch Python file.
import tifffile as tf   # External download
import skimage.exposure as exposure
from scipy import signal,ndimage
import scipy as sci
import time
from datetime import datetime
from alive_progress import alive_bar   # External download

#import pandas as pd
#from sklearn.preprocessing import StandardScalar
#from sklearn.decomposition import PCA

plt.close('all')
plt.rc('figure', max_open_warning = 0)   # Removes 20 plot limit warning  
now = datetime.now()
initial_time = now.strftime('%H:%M:%S')
print('Initialisation Time:', initial_time)
print('')
st = time.time()



# Strict Surface Line (Red)
def surfaceDetect(image, thresh):
    image = sci.ndimage.median_filter(image, size=(7,7))   # Applies median filter to image.
    length = np.size(image, 1)   # Sets length of x-axis.
    surfaceTemp = np.zeros(length)   # Creates array of zeros length of x-axis.
    start = 5
    finish = image.shape[0] - 5
    buffer = 10   # Buffer is to make sure the surface is correct, once the next 10 points are validated, will return and mark the surface.
    for i in range(0, length):   # Count-loop for x-axis loop.
        count = 0
        for j in range(start, finish):   # X-axis range
            if image[j, int(i)] > thresh:   # If pixel-value in image is greater than threshold, move to next column.
                count += 1
            else:
                count = 0
            if count == buffer:
                surfaceTemp[i] = j - buffer
                break
     
# Now we want to go through a correction loop and find any locations that have no surface identified and slowly lower the requirements.
    x = np.where(surfaceTemp==0)[0]

    for i in enumerate(x):
        threshNew = thresh
        count = 0
        done = False
        bufferNew = int(buffer/2)
        while done == False:
            threshNew = threshNew - np.ceil(thresh/100)   # Lowering the threshold value of pixel intensity.
            #print('threshold is now {}'.format(threshNew))
            for j in range(start, finish):
                if image[j, int(i[1])] > threshNew:
                    count += 1
                else:
                    count = 0
                if count == bufferNew:
                    surfaceTemp[i[1]] = j - bufferNew
                    done = True
                    #print('new surface value found')
                    break
        
    windowLength = 3   # Higher value = Smoother surface line. windowLength = 3 offers the most precision.
    surface = signal.savgol_filter(surfaceTemp, windowLength, 2).astype(int)   # Uses a filter to turn the values into a smooth line. 
    return(surface)



def depthDetect(intdB, thresh, scale, surface, buffer, skip):    
    #intdB = 10*np.log(intensity)
    #kernel = np.ones((5,5),np.uint8)
    #intdB = dilateErode(intdB, kernel)
    intdB = sci.ndimage.gaussian_filter(intdB,9*int(scale),0)
    #plt.figure()
    #plt.imshow(intdB,cmap='binary')
    #plt.clim([170,200])
    #plt.figure()
    #plt.plot(intdB[:,350])
    count = 0
    for i in range(0, len(surface)): # Added line
        for j in range(surface[i],len(intdB)):
            if intdB[j]<thresh:
                count +=1
            else:
                count = 0
            if count == buffer:
                depth = j - buffer + skip
                break
            if j == len(intdB)-1:
                depth  = j - 100
    return(depth)



def AttCoeff(Inten, surface, surf, depth,axres, col):   # Currently only taking the first B-scan in the .tif        surface, surf, depth, 
    Int = Inten[0, surf:depth, col]   # [0, surf:depth, col] for only signal
    att = np.zeros(np.size(Int, 0)) #-1
    for i in range(0,(np.size(Int,0)-1)):
        SumInt = np.sum(Int[(i+1):np.size(Int,0)])
        if SumInt == 0:
            SumInt = SumInt + 1e-30   # Bandaid fix to stop inf
        att[i] = (1/(2*axres)) * np.log(1 + Int[i]/SumInt)
     
    #plt.figure()
    #plt.plot(np.arange(0, np.size(att)), att)   # Plots Att. Coefficients's for a single A-scan for a selected column on a B-scan
    #plt.title('Attenuation Coefficients of a Single A-scan')
    #plt.xlabel('A-scan Depth [px]')
    #plt.ylabel('Attenuation Coefficient [mm^-1]')
    #plt.show()
    #print(att)
    avgatt = np.mean(att)   # Average Attenuation Coefficient
    return(att, avgatt, Int)



# IMPORT .TIF FILE:
image = tf.imread(r'D:\OCT ART Project\850 nm System\signature\output_a1\output_a1.tif')
im = exposure.rescale_intensity(image, in_range = 'image', out_range = (0, 255)).astype(np.uint8)   # Normalises tif.
del image   # Clearing Memory
print('Data Loaded.')
print('Processing...')



' ~ Individual B-scan Histogram ~ '

BscanNum = 1   # Selection of the B-scan you want analysed

Surface = surfaceDetect(im[BscanNum, 0:im.shape[1], 0:im.shape[2]], 100) # Plotting the surface of B-scans using the surfaceDetect function
#print('B-scan Surface:', Surface)



depthList = []   # Creating an empty list to append data into
for n in range(0, im.shape[2]):   # 'n' acts as a varying A-scan selection
    Depth = depthDetect(im[BscanNum,:,n], 50, 1, Surface, 10, 0) # Attenuation Depth of a single A-scan
    depthList.append(Depth)   # Appending data into list each loop
    #print('B-scan Depth:', Depth)



AttArray = np.zeros((im.shape[1],im.shape[2]))   # Creates an empty array filled with zeroes to store data into
for g in range(0, im.shape[2]):   # Looping the range of a B-scan
    IndividualAttCo, AvgAttCo, IntVal = AttCoeff(Inten=im, surface=Surface, surf=Surface[g], depth=depthList[g], axres=7e-3, col=g)   # Using the AttCoeff function
    #print(IndividualAttCo)
    AttArray[:np.size(IndividualAttCo),g] = IndividualAttCo   # Appending the attenuation coefficients at each pixel into an organised array
    #print('Average Attenuation Coefficient:', round(AvgAttCo, 5), 'mm^-1')



x = np.arange(0,im.shape[2],1)   # Width of B-scan for plotting
plt.imshow(im[BscanNum,:,:],cmap='gray')   # Plotting B-scan
line1 = plt.plot(x,Surface,'-', color='red', linewidth=1)   # Plotting surface line
line2 = plt.plot(x,depthList,'--', color='red', linewidth=0.5)   # Plotting sub-surface depth
plt.title('B-scan w/ Surface & Depth')
#plt.axis('off')
plt.show()



AttArray1D = AttArray.flatten()   # Converts the 2D array into 1D
steps = np.arange(0.001, 20, 0.1)   # Evenly spaced steps for histogram, zeroes not included as these will be from the empty matrix
plt.hist(AttArray1D, steps)   # Plotting histogram
plt.title('Histogram of Attenuation Coefficient Values (B-scan)')
plt.xlabel('Attenuation Coefficient [mm^-1]')
plt.ylabel('Frequency/Amount')
plt.show()




' ~ Full C-scan Histogram - Takes ~4.5 hours to compute ~ '

# AttArray = np.zeros((im.shape[1],im.shape[2]*im.shape[0]))   # Creates an empty array filled with zeroes to store data into
# #AttList = []
# with alive_bar(im.shape[0]) as bar:
#     for Bscan in range(0, im.shape[0]):
#         Surface = surfaceDetect(im[Bscan, 0:im.shape[1], 0:im.shape[2]], 70) # Plotting the surface of B-scans using the surfaceDetect function
#         #print('B-scan Surface:', Surface)
#         bar()
        
        
        
#         depthList = []   # Creating an empty list to append data into
#         for Ascan in range(0, im.shape[2]):   # 'n' acts as a varying A-scan selection
#             Depth = depthDetect(im[Bscan,:,Ascan], 50, 1, Surface, 10, 0) # Attenuation Depth of a single A-scan
#             depthList.append(Depth)   # Appending data into list each loop
#             #print('B-scan Depth:', Depth)
        
        
        
#         for g in range(0, im.shape[2]):   # Looping the range of a B-scan
#             IndividualAttCo, AvgAttCo, IntVal = AttCoeff(Inten=im, surface=Surface, surf=Surface[g], depth=depthList[g], axres=7e-3, col=g)   # Using the AttCoeff function
#             #AttList.append(IndividualAttCo)
#             AttArray[:np.size(IndividualAttCo),Ascan] = IndividualAttCo   # Appending the attenuation coefficients at each pixel into an organised array
#             #print('Average Attenuation Coefficient:', round(AvgAttCo, 5), 'mm^-1')



# # x = np.arange(0,im.shape[2],1)   # Width of B-scan for plotting
# # plt.imshow(im[BscanNum,:,:],cmap='gray')   # Plotting B-scan
# # line1 = plt.plot(x,Surface,'-', color='red', linewidth=1)   # Plotting surface line
# # line2 = plt.plot(x,Surface+depthList,'--', color='red', linewidth=0.5)   # Plotting sub-surface depth
# # plt.title('B-scan w/ Surface & Depth')
# # #plt.axis('off')
# # plt.show()


# #AttArrayFromList = np.array(AttList)
# #AttArray1D = AttArrayFromList.flatten()
# AttArray1D = AttArray.flatten()   # Converts the 2D array into 1D
# steps = np.arange(0.001, 20, 0.1)   # Evenly spaced steps for histogram, zeroes not included as these will be from the empty matrix
# plt.hist(AttArray1D, steps)   # Plotting histogram
# plt.title('Histogram of Attenuation Coefficient Values (B-scan)')
# plt.xlabel('Attenuation Coefficient [mm^-1]')
# plt.ylabel('Frequency/Amount')
# plt.show()



' ~ Principal Component Analysis (Currently not working) ~ '

# data = AttArray

# data_2d = data.reshape(-1, 1)

# # Apply PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(data_2d)

# # Print the explained variance ratio
# print("Explained variance ratio:", pca.explained_variance_ratio_)




print('')
elapsed_time = time.time() - st
print('Execution Time:', time.strftime('%H:%M:%S', time.gmtime(elapsed_time)), '(H:M:S)')


