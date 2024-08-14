# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:33:33 2023

@author: Hendrik
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'D:\2023\Artwork Data\840 nm System (Lumedica)\Rembrandt\20220607-155252\1 Images\B-scan Test.png')
plt.imshow(image)

# Convert to Grey
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Performing Binary Thresholding
kernel_size = 3
ret,thresh = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY)  # 200, 255

# Finding Contours 
cntrs = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

# Drawing Contours
radius = 2
colour = (255,30,30)
cv2.drawContours(image, cntrs, -1, colour, radius)
plt.imshow(image)