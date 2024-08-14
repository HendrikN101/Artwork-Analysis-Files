# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:37:59 2023

@author: User
"""

import numpy as np
import tifffile as tf   # External download
import skimage.exposure as exposure
import matplotlib.pyplot as plt



image = tf.imread(r'D:\2023 Scans\TIFS\Rembrandt Signature\Sign5.tif')   # Imports tif.
im = exposure.rescale_intensity(image, in_range = 'image', out_range = (0, 255)).astype(np.uint8)   # Normalises tif.

BscanNum = 310   # Select B-scan

stackedImages = np.array(())   # Creates empty array.
stackedImages = np.append(stackedImages, im)   # Appends the image data into the array.

finalImage = stackedImages.reshape(int(np.sum(im.shape[0])),int(im.shape[1]),np.size(im,2))   # Reshapes the data in array.

projection = np.sum(finalImage[:,:,:],1)   # Forms an en-face image of the C-scan image data.
plt.figure()
plt.axis('off')
plt.imshow(projection, cmap='gray')
x = np.arange(im.shape[2])   # For line plotting.
y = [BscanNum for i in range(0,im.shape[2])]   # ""
plt.plot(x, y, '--', color='red', linewidth=1)
plt.title('En-Face w/ B-scan Location')
plt.show()
del finalImage   # Clearing Memory.
del stackedImages


Bscan = im[BscanNum,:,:]   # Plotting B-scan
# plt.imshow(Int, cmap='gray')
# plt.title('B-scan')
# plt.show()


Int = np.abs(Bscan)
delta = 5e-3   # Pixel depth.
att = (1/(2*delta))*np.log(Int/(np.cumsum(Int[::-1],axis=0)[::-1]-Int) + 1)   # Attentuation coefficient equation.
# plt.imshow(att, vmin=0, vmax=4, cmap='gray')
# plt.title('Attenuation Coefficient Map')
# plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(Int, cmap='gray')
ax1.set_title('Original B-scan')
ax2.imshow(att, vmin=0, vmax=5, cmap='gray')
ax2.set_title('Att. Coefficient Map')

