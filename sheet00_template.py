import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


# Authors: Seyed Arash Safavi
# Laleh Samadfam

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # set image path
    img_path = 'bonn.png'
    # 2a: read and display the image
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    height, width = img.shape[0], img.shape[1]
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation
    img_cpy = img.copy()
    half_intensity_arr = img_gray * .5
    for y in range(0, height):
        for x in range(0, width):
            blue, green, red = img_cpy[y, x]
            blue = max(blue - half_intensity_arr[y][x], 0)
            green = max(green - half_intensity_arr[y][x], 0)
            red = max(red - half_intensity_arr[y][x], 0)
            img_cpy[y, x] = [blue, green, red]
    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perfom the operation above
    img_cpy = img.copy()
    img_cpy = np.uint8(np.maximum(img - np.expand_dims(half_intensity_arr, 2), 0))

    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)

    # 2e: Extract the center patch and place randomly in the image
    img_cpy = img.copy()
    patch_square_length = 16
    img_patch = np.zeros((patch_square_length, patch_square_length, 3), dtype=np.uint8)
    image_center_x = width // 2
    image_center_y = height // 2
    for y in range(patch_square_length):
        for x in range(patch_square_length):
            img_patch[y][x] = img_cpy[image_center_y - (patch_square_length // 2) + y][
                image_center_x - (patch_square_length // 2) + x]
    display_image('2 - e - Center Patch', img_patch)

    # Random location of the patch for placement
    img_cpy = img.copy()
    rand_coord = [0] * 2
    rand_coord[0] = max(0, randint(0, width) - (patch_square_length // 2))
    rand_coord[1] = max(0, randint(0, height) - (patch_square_length // 2))
    for y in range(patch_square_length):
        for x in range(patch_square_length):
            img_cpy[y + rand_coord[1]][x + rand_coord[0]] = img_patch[y][x]
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)

    # 2f: Draw random rectangles and ellipses
    img_cpy = img.copy()
    for index in range(10):
        rect_width = randint(20, 40)
        rect_height = randint(20, 40)
        rect_top_left_point = (randint(0, width - rect_width), randint(0, height - rect_height))
        rect_bottom_right_point = (rect_top_left_point[0] + rect_width, rect_top_left_point[1] + rect_height)
        color_rect = (randint(0, 256), randint(0, 256), randint(0, 256))
        color_ellipse = (randint(0, 256), randint(0, 256), randint(0, 256))
        ellipse_axis = randint(20, 40)
        ellipse_axes = (ellipse_axis, ellipse_axis)
        ellipse_center = (randint(ellipse_axis, width - ellipse_axis), randint(ellipse_axis, height - ellipse_axis))
        ellipse_rotation_angle = randint(0, 361)
        ellipse_start_angle = randint(0, 361)
        ellipse_end_angle = randint(10, 361)
        cv.rectangle(img_cpy, rect_top_left_point, rect_bottom_right_point, color_rect, -1)
        cv.ellipse(img_cpy, ellipse_center, ellipse_axes, ellipse_rotation_angle, ellipse_start_angle,
                   ellipse_end_angle,
                   color_ellipse, -1)
    display_image('2 - f - Rectangles and Ellipses', img_cpy)

    # destroy all windows
    cv.destroyAllWindows()
