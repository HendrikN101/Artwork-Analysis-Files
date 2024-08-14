# Importing Libraries
import numpy as np
import scipy as sci
from scipy import signal



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
            threshNew = threshNew - 1   # Lowering the threshold value of pixel intensity.
            #print('threshold is now {}'.format(threshNew))
            for j in range(start, finish):
                if image[j, int(i[1])] > threshNew:
                    count += 1
                else:
                    count = 0
                if count == bufferNew:
                    surfaceTemp[i[1]] = j - bufferNew
                    done = True
                    break # remove
                    # if surfaceTemp[i[1]] <= 20:   # The following process is to interpolate the surface in cases where the intensity is too low.
                    #     gradient = surfaceTemp[i[1]]-surfaceTemp[i[1]-3]
                    #     if gradient >= 1:
                    #         surfaceTemp[i[1]] = surfaceTemp[i[1]-1]-1
                    #         done = True
                    #         break
                    #     elif gradient <= 1:
                    #         surfaceTemp[i[1]] = surfaceTemp[i[1]-1]+1
                    #         done = True
                    #         break
                    #     else:
                    #         surfaceTemp[i[1]] = surfaceTemp[i[1]-1]
                    #         done = True
                    #         break
        
    windowLength = 5 #5   # Higher value = Smoother surface line. windowLength = 3 offers the most precision.
    surface = signal.savgol_filter(surfaceTemp, windowLength, 2).astype(int)   # Uses a filter to turn the values into a smooth line. 
    return(surface)


# Smooth Surface Line (Blue)
def surfaceDetect2(image, thresh):
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
            threshNew = threshNew - 1   # Lowering the threshold value of pixel intensity.
            #print('threshold is now {}'.format(threshNew))
            for j in range(start, finish):
                if image[j, int(i[1])] > threshNew:
                    count += 1
                else:
                    count = 0
                if count == bufferNew:
                    surfaceTemp[i[1]] = j - bufferNew
                    done = True
                    if surfaceTemp[i[1]] <= 20:   # The following process is to interpolate the surface in cases where the intensity is too low.
                        gradient = surfaceTemp[i[1]]-surfaceTemp[i[1]-3]
                        if gradient > 0:
                            surfaceTemp[i[1]] = surfaceTemp[i[1]-1]-1
                            done = True
                            break
                        elif gradient < 0:
                            surfaceTemp[i[1]] = surfaceTemp[i[1]-1]+1
                            done = True
                            break
                        else:
                            surfaceTemp[i[1]] = surfaceTemp[i[1]-1]
                            done = True
                            break
        
    windowLength = 21 #21
    surface = signal.savgol_filter(surfaceTemp, windowLength, 2).astype(int)   # Uses a filter to turn the values into a smooth line. 
    return(surface)


