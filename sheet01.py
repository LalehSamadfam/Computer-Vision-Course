import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys

if __name__ == '__main__':
    img_path = sys.argv[1]


def my_integral(img):
    temp_img = cv.copyMakeBorder(img, 1, 0, 1, 0, cv.BORDER_CONSTANT, None,
                                 0)  # add a border of zeros to top and left of image

    for i in range(1, temp_img.shape[0]):  # rows
        for j in range(1, temp_img.shape[1]):  # columns
            temp_img[i, j] = temp_img[i - 1, j] + temp_img[i, j - 1] + temp_img[i, j] - temp_img[i - 1, j - 1]

    integral_img = temp_img[1:, 1:]  # removing the border
    return integral_img


def mean_integral_i(img):
    mean = 0
    for i in range(img.shape[0]):  # rows
        for j in range(img.shape[1]):  # columns
            mean += img[i, j]
    mean /= img.size
    return mean


def mean_integral_ii(img):
    return cv.integral(img)[-1, -1] / img.size


def mean_integral_iii(img):
    integral_img = my_integral(img)
    return integral_img[-1, -1] / integral_img.size


#extracts a randomly placed rectangle with a given size form image
def get_patch(img, patch_height, patch_width):
    height, width = img.shape[0], img.shape[1]
    row = random.randint(1, height - patch_height)
    col = random.randint(1, width - patch_width)
    return img[row:row + patch_height, col + col + patch_width]


#    =========================================================================    
#    ==================== Task 1 =================================
#    =========================================================================    
print('Task 1:');

img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # read image

# a
integral_img = my_integral(img)
cv.imshow('integral image', integral_img)

# b
# using sum of image values
mean_value_i = mean_integral_i(img)

# using cv.integral
mean_value_ii = mean_integral_ii(img)

# using integral image of our own
mean_value_iii = mean_integral_iii(img)

# c
for i in range(10):
    patch = get_patch(img,100, 100)

    t_start = time.process_time()
    mean_integral_i(patch)
    t_end = time.process_time()
    print('processtime = ' + (t_end - t_start))

    t_start = time.process_time()
    mean_integral_ii(patch)
    t_end = time.process_time()
    print('processtime = ' + (t_end - t_start))

    t_start = time.process_time()
    mean_integral_iii(patch)
    t_end = time.process_time()
    print('processtime = ' + (t_end - t_start))

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
