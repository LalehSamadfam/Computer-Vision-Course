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
    sum = 0
    for i in range(img.shape[0]):  # rows
        for j in range(img.shape[1]):  # columns
            sum += img[i, j]
    mean = sum / img.size
    return mean


def mean_integral_ii(img):
    return cv.integral(img)[-1, -1] / img.size


def mean_integral_iii(img):
    integral_img = my_integral(img)
    return integral_img[-1, -1] / integral_img.size


# extracts a randomly placed rectangle with a given size form image
def get_patch(img, patch_height, patch_width):
    height, width = img.shape[0], img.shape[1]
    row = random.randint(1, height - patch_height)
    col = random.randint(1, width - patch_width)
    return img[row:row + patch_height, col: col + patch_width]


def hist_eq(img):
    width, height, size = img.shape[0], img.shape[1], img.size
    # creating histogram of an image
    hist = np.zeros(256)
    for i in range(width):
        for j in range(height):
            hist[img[i, j]] += 1
    # creating cumulitive histogram of image
    cumulitive_hist = np.zeros(256)
    cumulitive_hist[0] = hist[0]
    for i in range(1, hist.size):
        cumulitive_hist[i] = cumulitive_hist[i - 1] + hist[i]
    # histogram equalize image
    for i in range(width):
        for j in range(height):
            img[i, j] = 255 / size * cumulitive_hist[img[i, j]]
    return img


# compute maximum pixelwise error of two images
def max_err(src, res):
    err = 0
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i, j] - res[i, j] > err:
                err = src[i, j] - res[i, j]
    return err


def salt_pepper(img, p):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_value = np.random.choice([img[i, j], 0, 256], 1, [1 - p, p / 2, p / 2])
            img[i,j] = random_value[0]
    return img


def update_mean_gray_err(src, res):
    return mean_integral_ii(src) - mean_integral_ii(res)


def gaussian_kernel(sigma):
    kernel_size = int(6 * sigma)
    kernel = np.zeros([kernel_size, kernel_size])
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.e ** (-(i ** 2 + j ** 2) / (2 * sigma ** 2))
    return kernel


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
input('press any key to continue')

# b
# using sum of image values
mean_value_i = mean_integral_i(img)

# using cv.integral
mean_value_ii = mean_integral_ii(img)

# using integral image of our own
mean_value_iii = mean_integral_iii(img)

print(mean_integral_i(img), mean_integral_ii(img), mean_integral_iii(img))

# c
for i in range(10):
    patch = get_patch(img, 10, 10)
    t_start = time.process_time()
    mean_integral_i(patch)
    t_end = time.process_time()
 #   print('patch', i, 'mean integral i process time = ', (t_end - t_start))

    t_start = time.process_time()
    mean_integral_ii(patch)
    t_end = time.process_time()
#    print('patch', i, 'mean integral ii process time = ', (t_end - t_start))

    t_start = time.process_time()
    mean_integral_iii(patch)
    t_end = time.process_time()
#    print('patch', i, 'mean integral iii process time = ', (t_end - t_start))


#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================    
print('Task 2:');

# a
cv_img_hist_eq = cv.equalizeHist(img)
cv.imshow('cv histogram equalized', cv_img_hist_eq)
input('press any key to continue')

my_img_hist_eq = hist_eq(img)
cv.imshow('my histogram qualized', my_img_hist_eq)
input('press any key to continue')
print(max_err(cv_img_hist_eq, my_img_hist_eq))

#    =========================================================================
#    ==================== Task 4 =================================
#    ========================================================================= 

print('Task 4:');
cv.imshow('bonn.png', img)
input('press any key to continue')

sigma = 2 * np.sqrt(2)
# a
cv_blurred = cv.GaussianBlur(img, (0, 0), 2 * np.sqrt(2))
cv.imshow('blurred with cv gaussian blur function', cv_blurred)
input('press any key to continue')
# b
cv_kernel = cv.getGaussianKernel(int(6*sigma), sigma)
cv_gaus_blurred = cv.filter2D(img, -1, cv_kernel)
cv.imshow('blurred with cv gaussian kernel', cv_gaus_blurred)
input('press any key to continue')

# c
my_kernel = gaussian_kernel(sigma)
my_gaus_blurred = cv.filter2D(img, -1, my_kernel)
cv.imshow('blurred with self implemented gaussian kernel', my_gaus_blurred)
input('press any key to continue')

# calculation of maximum pixel wise error of three pairs
print('max error for a, b = ', max_err(cv_blurred, cv_gaus_blurred))
print('max error for a, c = ', max_err(cv_blurred, my_gaus_blurred))
print('max error for b, c = ', max_err(cv_gaus_blurred, my_gaus_blurred))

#    =========================================================================
#    ==================== Task 5 =================================
#    =========================================================================    
print('Task 5:');

cv.imshow('bonn.png', img)
input('press any key to continue')

once_convolved = cv.GaussianBlur(img, (0, 0), 2)
two_step_convolved = cv.GaussianBlur(once_convolved, (0, 0), 2)
cv.imshow('twice convolved with kernel size 2', two_step_convolved)
input('press any key to continue')

one_step_convolve = cv.GaussianBlur(img, (0, 0), 2 * np.sqrt(2))
cv.imshow('once convolved with kernel size 2*sqrt(2)', one_step_convolve)
input('press any key to continue')

print('max error for convolving twice with sigma = 2 and once with 2*sqrt(2) is ', max_err(two_step_convolved,
                                                                                           one_step_convolve))

#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================    
print('Task 7:');

noisy_img = salt_pepper(img, 0.3)
cv.imshow('noisy image', noisy_img)
input('press any key to continue')

#a
gaussian_errors = np.zeros(5)

#b
median_errors = np.zeros(5)

#c
bilateral_errors = np.zeros(5)

for k in [1, 3, 5, 7, 9]:
    # a
    gaussian_filtered = cv.GaussianBlur(noisy_img, (k, k), k/6)
    gaussian_errors[k // 2] = update_mean_gray_err(img, gaussian_filtered)

    # b
    median_filtered = cv.medianBlur(img, k)
    median_errors[k // 2] = update_mean_gray_err(img, median_filtered)

    # c
    bilateral_filtered = cv.bilateralFilter(img, k, k/6, k/6) #TODO check bilateral filter sigma
    bilateral_errors[k // 2] = update_mean_gray_err(img, bilateral_filtered)


#a
g_size = np.min(gaussian_errors)

gaussian_filtered = cv.GaussianBlur(noisy_img, (g_size,g_size), g_size/6)
cv.imshow('problem 7, gaussian blurred', gaussian_filtered)
input('press any key to continue')

#b
m_size = np.min(median_errors)
median_filtered = cv.medianBlur(img, m_size)
cv.imshow('problem 7, median filtered', median_filtered)
input('press any key to continue')

#c
b_size = np.min(bilateral_errors)
bilateral_filtered = cv.bilateralFilter(img, k, k/6, k/6) #TODO
cv.imshow('problem 7, bilateral filtered', bilateral_filtered)
input('press any key to continue')


#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================    
print('Task 8:');

kernel_a = cv.UMat(np.array( ([0.0113, 0.0838, 0.0113],[0.0838, 0.6193, 0.0838], [0.0113, 0.0838, 0.0113]), dtype=np.uint8))
kernel_b = cv.UMat(np.array( ([-0.8984, 0.1472, 1.1410],[-1.9075, 0.1566, 2.1359], [-0.8659, 0.0573, 1.0337]), dtype=np.uint8))


#a
a_filtered = cv.filter2D(img, -1,  kernel_a)
b_filtered = cv.filter2D(img, -1,  kernel_b)

#b

W_a, U_a, V_a = cv.SVDecomp(kernel_a)
W_b, U_b, V_b = cv.SVDecomp(kernel_b)
