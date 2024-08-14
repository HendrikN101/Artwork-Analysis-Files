# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:39:42 2023

@author: Hendrik
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


# IMPORT IMAGE
img = cv2.imread(r'D:\2023\Artwork (New)\840 nm System (Lumedica)\Rembrandt\20220607-155252\1 Images\Test 2.png',0)[:,:]

plt.imshow(img, cmap='gray')
plt.title('Original Image (Greyscale)')
plt.show()

kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])   # Sharpening Kernel
kernel_edge_Gx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])   # Gradient Kernel (Gx)
kernel_edge_Gy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])   # Gradient Kernel (Gy)


def fourier_masker_vert(image, i):   # FOR LOCATING VERTICAL LINES
    magnitude_spectrum = np.fft.fftshift(np.fft.fft2(img))   # Calculating 2D FFT of Image
    #magnitude_spectrum[math.ceil(img[:,0].size/2.02):math.ceil(img[:,0].size/1.99), math.ceil(img[0].size/2.01):math.ceil(img[0].size/1.99)] = i   # Removing DC Component
    magnitude_spectrum[:math.ceil(img[:,0].size/2.01), math.floor(img[0].size/2.11):math.ceil(img[0].size/1.89)] = i   # Masking Spectrum
    magnitude_spectrum[-math.floor(img[:,0].size/2.01):, math.floor(img[0].size/2.11):math.ceil(img[0].size/1.89)] = i   # Masking Spectrum
    
    # MASKED 2D FT
    plt.imshow(np.log(abs(magnitude_spectrum)), cmap='gray')
    plt.title('Masked 2D Fourier Transform')
    plt.show()
    
    # TRANSFORMED IMAGE
    plt.imshow(abs(np.fft.ifft2(magnitude_spectrum)), cmap='gray')
    plt.title('Transformed Image (Vert. Emph.)')
    plt.show()
    
    # SHARPENED TRANSFORMATION
    sharpened_vert = cv2.filter2D(abs(np.fft.ifft2(magnitude_spectrum)), -1, kernel)
    plt.imshow(sharpened_vert, cmap='gray')
    plt.title('Transformed Image (Vert., Sharp.)')
    plt.show()
    
    # GRADIENT TRANSFORMATION
    edge_vert = cv2.filter2D(abs(np.fft.ifft2(magnitude_spectrum)), -1, kernel_edge_Gx)
    plt.imshow(edge_vert, cmap='gray')
    plt.title('Transformed Image (Vert., Edge)')
    plt.show()
    
fourier_masker_vert(img, 1)


def fourier_masker_horiz(image, i):   # FOR LOCATING HORIZONTAL LINES
    magnitude_spectrum = np.fft.fftshift(np.fft.fft2(img))
    #magnitude_spectrum[math.floor(img[:,0].size/2.001):math.ceil(img[:,0].size/1.999), math.floor(img[0].size/2.001):math.ceil(img[0].size/1.999)] = i   # Removing DC Component
    magnitude_spectrum[math.floor(img[:,0].size/2.11):math.ceil(img[:,0].size/1.89), :math.ceil(img[0].size/2.01)] = i   # Masking Spectrum
    magnitude_spectrum[math.floor(img[:,0].size/2.11):math.ceil(img[:,0].size/1.89), -math.floor(img[0].size/2.01):] = i   # Masking Spectrum
    
    # MASKED 2D FT
    plt.imshow(np.log(abs(magnitude_spectrum)), cmap='gray')
    plt.title('Masked 2D Fourier Transform')
    plt.show()
    
    # TRANSFORMED IMAGE
    transformed_horiz = abs(np.fft.ifft2(magnitude_spectrum))
    plt.imshow(transformed_horiz, cmap='gray')
    plt.title('Transformed Image (Horiz. Emph.)')
    plt.show()
    
    # SHARPENED TRANSFORMATION
    sharpened_horiz = cv2.filter2D(abs(np.fft.ifft2(magnitude_spectrum)), -1, kernel)
    plt.imshow(sharpened_horiz, cmap='gray')
    plt.title('Transformed Image (Horiz., Sharp.)')
    plt.show()
    
    # GRADIENT TRANSFORMATION
    edge_horiz = cv2.filter2D(abs(np.fft.ifft2(magnitude_spectrum)), -1, kernel_edge_Gy)
    plt.imshow(edge_horiz, cmap='gray')
    plt.title('Transformed Image (Horiz., Edge)')
    plt.show()
    
fourier_masker_horiz(img, 1)
