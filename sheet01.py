import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys


if __name__ == '__main__':
    img_path = sys.argv[1]


#    =========================================================================    
#    ==================== Task 1 =================================
#    =========================================================================    
print('Task 1:');
#a
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
integral_img = cv.copyMakeBorder(img, 1, 0, 1, 0, cv.BORDER_CONSTANT, None, 0) #add a border of zeros to top and left of image

for i in range(1, img.shape[0]): #rows
    for j in range(1, img.shape[1]): #columns
        integral_img[i, j] = integral_img[i - 1, j] + integral_img[i, j - 1] + img[i,j] - img[i - 1, j - 1]

integral_img = integral_img[ 1:, 1:] #removing the border

#b
#using sum of image values
mean_value_1 = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        mean_value_1 += img[i,j]
mean_value_1 /= img.size

#using cv.integral
mean_value_2 = cv.integral(img)[-1, -1] / img.size

#using integral image of our own
mean_value_3 = integral_img[-1, -1]/integral_img.size








#    =========================================================================    
#    ==================== Task 2 =================================
#    =========================================================================    
print('Task 2:');





#    =========================================================================    
#    ==================== Task 4 =================================
#    =========================================================================    
print('Task 4:');





#    =========================================================================    
#    ==================== Task 6 =================================
#    =========================================================================    
print('Task 6:');





#    =========================================================================    
#    ==================== Task 7 =================================
#    =========================================================================    
print('Task 7:');





#    =========================================================================    
#    ==================== Task 8 =================================
#    =========================================================================    
print('Task 8:');



