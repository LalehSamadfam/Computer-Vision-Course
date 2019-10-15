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
    return img[row:row + patch_height, col : col + patch_width]

def hist_eq(img):
    width, height, size = img.shape[0], img.shape[1], img.size
    #creating histogram of an image
    hist = np.zeros(256)
    for i in range(width):
        for j in range(height):
            hist[img[i,j]] += 1
    #creating cumulitive histogram of image
    cumulitive_hist = np.zeros(256)
    cumulitive_hist[0] = hist[0]
    for i in range(1, hist):
        cumulitive_hist[i] = cumulitive_hist[i - 1] + hist[i]
    #histogram equalize image
    for i in range(width):
        for j in range(height):
            img[i,j] = 255 / size * cumulitive_hist[img[i,j]]
    return img

def max_err(src, res):
    err = 0
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i,j] - res[i,j] > err:
                err = src[i,j] - res[i,j]
    return err
#    =========================================================================
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # read image
#    =========================================================================

#    =========================================================================    
#    ==================== Task 1 =================================
#    =========================================================================    

 print('Task 1:');
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
     patch = get_patch(img,10, 10)

     t_start = time.process_time()
     mean_integral_i(patch)
     t_end = time.process_time()
     print('patch', i, 'mean integral i process time = ' , (t_end - t_start))

     t_start = time.process_time()
     mean_integral_ii(patch)
     t_end = time.process_time()
     print('patch', i, 'mean integral ii process time = ' , (t_end - t_start))

     t_start = time.process_time()
     mean_integral_iii(patch)
     t_end = time.process_time()
     print('patch', i, 'mean integral iii process time = ' , (t_end - t_start))

#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================    
print('Task 2:');

#a
cv_img_hist_eq = cv.equalizeHist(img)
cv.imshow('cv histogram equalized', cv_img_hist_eq)

my_img_hist_eq = hist_eq(img)
cv.imshow('my histogram qualized', my_img_hist_eq)

print(max_err(cv_img_hist_eq, my_img_hist_eq))


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
