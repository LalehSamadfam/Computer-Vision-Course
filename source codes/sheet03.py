import numpy as np
import cv2 as cv
import random


##############################################
#     Task 1        ##########################
##############################################

def display_lines(lines, image, header):
    line_size = max(image.shape[0], image.shape[1])
    for l in lines:
        rho, theta = l[0][0], l[0][1]
        x0 = rho * np.cos(theta)  # x0 is a point that we know is located on the line.
        y0 = rho * np.sin(theta)
        x1 = int(x0 - np.sin(theta) * line_size)
        y1 = int(y0 + np.cos(theta) * line_size)
        x2 = int(x0 + np.sin(theta) * line_size)
        y2 = int(y0 - np.cos(theta) * line_size)
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv.imshow(header, image)
    cv.waitKey(0)

def my_display_lines(lines, image, header):
    line_size = max(image.shape[0], image.shape[1])
    for l in lines:
        theta, rho = l
        x0 = rho * np.cos(theta)  # x0 is a point that we know is located on the line.
        y0 = rho * np.sin(theta)
        x1 = int(x0 - np.sin(theta) * line_size)
        y1 = int(y0 + np.cos(theta) * line_size)
        x2 = int(x0 + np.sin(theta) * line_size)
        y2 = int(y0 - np.cos(theta) * line_size)
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv.imshow(header, image)
    cv.waitKey(0)

def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('images/shapes.png')
    img_copy = img.copy()
    edges = cv.Canny(img, 150, 300, None, 3)
    lines = cv.HoughLines(edges, 1, 2 * np.pi/180, 50)
    print(lines)
    display_lines(lines, img_copy, 'Task 1 a')
    cv.destroyAllWindows()


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    for cell in np.argwhere(img_edges > 200):
            x, y = cell
            for step in range(int(180/theta_step_sz)):
                theta = step * theta_step_sz
                d = int(x * np.cos(theta) + y * np.sin(theta))
                accumulator[step,  d] += 1
    detected_lines = np.argwhere(accumulator > threshold)
    print(detected_lines)
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('images/shapes.png')
    img_copy = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert the image into grayscale
    edges = cv.Canny(img_gray, 150, 300, None, 3)  # detect the edges
    lines, accumulator = myHoughLines(edges, 1, 2, 50)
    my_display_lines(lines, img_copy, 'task 1b')

##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    theta_res = None # set the resolution of theta
    d_res = None # set the distance resolution
    #_, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    # ....

    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...

        # update clusters' centers and check for convergence
        # ...

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = None  # construct the D matrix
    W = None  # construct the W matrix
    '''
    ...
    your code ...
    ...
    '''


##############################################
##############################################
##############################################

if __name__ == "__main__":
    task_1_a()
    task_1_b()
   # task_2()
   # task_3_a()
   # task_3_b()
   # task_3_c()
   # task_4_a()

