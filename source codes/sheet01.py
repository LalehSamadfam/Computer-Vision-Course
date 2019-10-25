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
    integral_img = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=int)
    for i in range(1, integral_img.shape[0]):  # rows
        for j in range(1, integral_img.shape[1]):  # columns
            integral_img[i, j] = int(integral_img[i - 1, j]) + int(integral_img[i, j - 1]) + int(temp_img[i, j]) -\
                                 int(integral_img[i - 1, j - 1])
    integral_img = integral_img[1:, 1:]  # removing the border
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
            if int(src[i, j]) - int(res[i, j]) > err:
                err = int(src[i, j]) - int(res[i, j])
    return err


def salt_pepper(img, p):
    img_copy = img.copy()
    plain = np.ones(100 - p)
    peper = np.zeros(p // 2 - 1)
    salt = 255 * np.ones(p // 2 + 1)
    dice = np.append(plain, peper)
    dice = np.append(dice, salt)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_value = np.random.choice(dice)
            if(random_value != 1):
               img_copy[i,j] = random_value
        print(img_copy[i,j])
    return img_copy


def update_mean_gray_err(src, res):
    return mean_integral_ii(src) - mean_integral_ii(res)


def gaussian_kernel(sigma):
    kernel_size = int(6*sigma) - 1
    kernel = np.zeros([kernel_size, kernel_size])
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.e ** (-(i ** 2 + j ** 2) / (2 * sigma ** 2))
    return kernel

def cast_to_image(integral_img):
    casted = np.zeros((integral_img.shape[0], integral_img.shape[1]))
    for i in range(0, integral_img.shape[0]):
        for j in range(0, integral_img.shape[1]):
            if integral_img[i, j] > 256:
                casted[i, j] = 256
            else:
                casted[i,j] = integral_img[i,j]
    return casted


#    =========================================================================
img = cv.imread(img_path, 0)  # read image
#    =========================================================================

#    =========================================================================    
#    ==================== Task 1 =================================
#    =========================================================================

print('Task 1:');

# a
integral_img = my_integral(img)
casted = cast_to_image(integral_img)
cv.imshow('the values above 256 are all whitend', casted)
input('press any key to continue')

# b
# using sum of image values
mean_value_i = mean_integral_i(img)
print('mean value i = ', mean_value_i)

# using cv.integral
mean_value_ii = mean_integral_ii(img)
print('mean value ii=', mean_value_ii)

# using integral image of our own
mean_value_iii = mean_integral_iii(img)
print('mean value iii = ', mean_value_iii)

# c
for i in range(10):
    patch = get_patch(img, 10, 10)
    t_start = time.process_time()
    mean_integral_i(patch)
    t_end = time.process_time()
    print('patch', i, 'mean integral i process time = ', (t_end - t_start))

    t_start = time.process_time()
    mean_integral_ii(patch)
    t_end = time.process_time()
    print('patch', i, 'mean integral ii process time = ', (t_end - t_start))

    t_start = time.process_time()
    mean_integral_iii(patch)
    t_end = time.process_time()
    print('patch', i, 'mean integral iii process time = ', (t_end - t_start))


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
print('maximum error for opencv histogram equalization and ours is =', max_err(cv_img_hist_eq, my_img_hist_eq))

#    =========================================================================
#    ==================== Task 4 =================================
#    ========================================================================= 

print('Task 4:');
cv.imshow('bonn.png', img)
input('press any key to continue')

sigma = 2 * np.sqrt(2)
kernel_size = int(6*sigma) - 1

# a
cv_blurred = cv.GaussianBlur(img, (kernel_size, kernel_size), 2 * np.sqrt(2))
cv.imshow('a: blurred with cv gaussian blur function', cv_blurred)
input('press any key to continue')

# b
cv_kernel = cv.getGaussianKernel(kernel_size, sigma)
cv_gaus_blurred = cv.filter2D(img, -1, cv_kernel)
cv.imshow('b: blurred with cv gaussian kernel', cv_gaus_blurred)
input('press any key to continue')

# c
my_kernel = gaussian_kernel(sigma)
my_gaus_blurred = cv.filter2D(img, -1, my_kernel)
cv.imshow('c: blurred with self implemented gaussian kernel', my_gaus_blurred)
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

noisy_img = salt_pepper(img, 30)
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
    median_errors[k // 2] = update_mean_gray_err(noisy_img, median_filtered)

    # c
    bilateral_filtered = cv.bilateralFilter(noisy_img, k, k/6, k/6) #TODO check bilateral filter sigma
    bilateral_errors[k // 2] = update_mean_gray_err(img, bilateral_filtered)


#a
g_size = 2 * np.argmin(gaussian_errors) + 1

gaussian_filtered = cv.GaussianBlur(noisy_img, (g_size,g_size), g_size/6)
cv.imshow('problem 7, gaussian blurred', gaussian_filtered)
input('press any key to continue')

#b
m_size = 2 * np.argmin(median_errors) + 1
median_filtered = cv.medianBlur(noisy_img, m_size)
cv.imshow('problem 7, median filtered', median_filtered)
input('press any key to continue')

#c
b_size = 2 * np.argmin(bilateral_errors) + 1
bilateral_filtered = cv.bilateralFilter(noisy_img, k, k/6, k/6) #TODO
cv.imshow('problem 7, bilateral filtered', bilateral_filtered)
input('press any key to continue')
#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================    

print('Task 8:');

kernel_a = cv.UMat(np.array([[0.0113, 0.0838, 0.0113],[0.0838, 0.6193, 0.0838], [0.0113, 0.0838, 0.0113]]))
kernel_b = cv.UMat(np.array([[-0.8984, 0.1472, 1.1410],[-1.9075, 0.1566, 2.1359], [-0.8659, 0.0573, 1.0337]]))

#a
a_filtered = cv.filter2D(img, -1, kernel_a)
cv.imshow('filtered with kernel a', a_filtered)
input('press any key to continue')

b_filtered = cv.filter2D(img, -1, kernel_b)
cv.imshow('filtered with kernel b', b_filtered)
input('press any key to continue')


#b
W_a, U_a, Vt_a = cv.SVDecomp(kernel_a)
rank_a = np.linalg.matrix_rank(kernel_a)

horiz_kern_a = np.array(U_a.get()[0]) * np.sqrt(W_a.get()[0][0]) + np.array(U_a.get()[1]) * np.sqrt(W_a.get()[1][0])
vert_kern_a = np.array(Vt_a.get()[0]) * np.sqrt(Vt_a.get()[0][0]) + np.array(Vt_a.get()[1]) * np.sqrt(W_a.get()[1][0])
h_filtered = cv.filter2D(img, -1, horiz_kern_a)
decomp_filtered_a = cv.filter2D(h_filtered, -1, vert_kern_a)
cv.imshow('filterd with kernel a compositions', decomp_filtered_a)
input('press any key to continue')


W_b, U_b, Vt_b = cv.SVDecomp(kernel_b)
horiz_kern_b = np.dot(U_b.get()[0], np.sqrt(W_b.get()[0][0]))
vert_kern_b = np.dot(Vt_b.get()[0], np.sqrt(W_b.get()[0][0]))
h_filtered = cv.filter2D(img, -1, horiz_kern_a)
decomp_filtered_b = cv.filter2D(h_filtered, -1, vert_kern_b)
cv.imshow('filtered with kernel b compositions', decomp_filtered_b)
input('press any key to continue')

#c

print('max error between section a, b for kernel a = ', max_err(decomp_filtered_a, a_filtered.get()))
print('max error between section a, b for kernel b = ', max_err(decomp_filtered_b, b_filtered.get()))
