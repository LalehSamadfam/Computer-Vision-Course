import cv2
import numpy as np
import time


def get_convolution_using_fourier_transform(image, kernel):
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, image_fft.shape)
    conv_fft = image_fft * kernel_fft
    fft_result = np.fft.ifft2(conv_fft)
    return fft_result


def task1():
    image = cv2.imread("../data/einstein.jpeg", 0)
    kernel = cv2.getGaussianKernel(7, 1)  # calculate kernel

    conv_result = cv2.filter2D(image, -1, kernel)  # calculate convolution of image and kernel
    fft_result = get_convolution_using_fourier_transform(image, kernel) #get_convolution_using_fourier_transform(image, kernel)

    cv2.imshow('conv', conv_result)
    input('press any key to continue..')

    cv2.imshow('fft', fft_result)
    input('press any key to continue..')

def sum_square_difference(image, template):
    return None


def normalized_cross_correlation(image, template):
    return None

"""""
def task2():
    image = cv2.imread("../data/lena.png", 0)
    template = cv2.imread("../data/eye.png", 0)

    result_ssd = sum_square_difference(image, template)
    result_ncc = normalized_cross_correlation(image, template)

    result_cv_sqdiff = None  # calculate using opencv
    result_cv_ncc = None  # calculate using opencv

    # draw rectangle around found location in all four results
    # show the results


def build_gaussian_pyramid_opencv(image, num_levels):
    return None


def build_gaussian_pyramid(image, num_levels, sigma):
    return None


def template_matching_multiple_scales(pyramid, template):
    return None


def task3():
    image = cv2.imread("../data/traffic.jpg", 0)
    template = cv2.imread("../data/template.jpg", 0)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 8)
    mine_pyramid = build_gaussian_pyramid(image, 8)

    # compare and print mean absolute difference at each level
    result = template_matching_multiple_scales(pyramid, template)

    # show result


def get_derivative_of_gaussian_kernel(size, sigma):
    return None, None


def task4():
    image = cv2.imread("../data/einstein.jpeg", 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = None  # convolve with kernel_x
    edges_y = None  # convolve with kernel_y

    magnitude = None  # compute edge magnitude
    direction = None  # compute edge direction

    cv2.imshow("Magnitude", magnitude)
    cv2.imshow("Direction", direction)


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    return None


def task5():
    image = cv2.imread("../data/traffic.jpg", 0)

    edges = None  # compute edges
    edge_function = None  # prepare edges for distance transform

    dist_transfom_mine = l2_distance_transform_2D(
        edge_function, positive_inf, negative_inf
    )
    dist_transfom_cv = None  # compute using opencv

    # compare and print mean absolute difference

"""
if __name__ == "__main__":
    task1()
   # task2()
    #task3()
    #task4()
    #task5()
