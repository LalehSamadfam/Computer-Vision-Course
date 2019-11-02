import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Authors:
# Lale Samadfam
# Seyed Arash Safavi


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
        # theta = step * 2 * np.pi/180
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
    lines = cv.HoughLines(edges, 1, 2 * np.pi / 180, 50)
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
        for step in range(int(180 / theta_step_sz)):
            theta = step * theta_step_sz * np.pi / 180
            d = int(x * np.sin(theta) + y * np.cos(theta) / d_resolution)
            accumulator[step, d] += 1
    detected_lines = np.argwhere(accumulator > threshold)
    fixer = np.concatenate((theta_step_sz * np.pi / 180 * np.ones((5, 1)), np.ones((5, 1))), axis=1)
    detected_lines = np.multiply(fixer, detected_lines)
    return detected_lines, accumulator


def iter_count(C, max_iter):
    X = C
    for n in range(max_iter):
        if abs(X) > 2.:
            return n
        X = X ** 2 + C
    return max_iter


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('images/shapes.png')
    img_copy = img.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert the image into grayscale
    edges = cv.Canny(img_gray, 150, 300, None, 3)  # detect the edges
    lines, accumulator = myHoughLines(edges, 1, 2, 50)
    plt.imshow(accumulator, cmap=cm.gray)
    plt.show()
    my_display_lines(lines, img_copy, 'task 1b')


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = None  # convert the image into grayscale
    edges = None  # detect the edges
    theta_res = None  # set the resolution of theta
    d_res = None  # set the distance resolution
    # _, accumulator = myHoughLines(edges, d_res, theta_res, 50)
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
    vertices = 8
    W = np.array([[0, 1, .2, 1, 0, 0, 0, 0],
                  [1, 0, .1, 0, 1, 0, 0, 0],
                  [.2, .1, 0, 1, 0, 1, .3, 0],
                  [1, 0, 1, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 0, 0, 1, 0],
                  [0, 0, .3, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0, 1, 0]])  # construct the W matrix
    D = np.zeros(shape=(vertices, vertices))  # construct the D matrix
    for i in range(vertices):
        D[i, i] = np.sum(W[i])

    L = D - W
    D_sqrt = np.sqrt(D)
    D_sqrt_inv = np.linalg.inv(D_sqrt)
    _, eigen_values, eigen_vectors = cv.eigen(np.matmul(np.matmul(D_sqrt_inv, L), D_sqrt_inv))
    y2 = np.dot(np.linalg.inv(D_sqrt), eigen_vectors[-2])
    print(y2)

    ##4_b
    print("Task 4 (b) ...")

    characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    C1 = set()
    C2 = set()
    for index, value in enumerate(y2):
        if value < 0:
            C1.add(characters[index])
        else:
            C2.add(characters[index])
    print("Cluster 1: ", C1)
    print("Cluster 2: ", C2)
    weight_sum = 0
    for index, weight in np.ndenumerate(W):
        if characters[index[0]] in C1 and \
                characters[index[1]] in C2:
            weight_sum += weight
    volume_1 = 0
    volume_2 = 0
    for index, char in enumerate(characters):
        if char in C1:
            volume_1 += np.sum(W[index])
        else:
            volume_2 += np.sum(W[index])

    cost = (weight_sum) / volume_1 + (weight_sum) / volume_2
    print("Cost: ", cost)


##############################################
##############################################
##############################################

if __name__ == "__main__":
    task_1_a()
    task_1_b()
    # # task_2()
    # task_3_a()
    # task_3_b()
    # task_3_c()
    task_4_a()
