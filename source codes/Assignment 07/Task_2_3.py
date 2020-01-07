import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def read_train_data(filepath):
    data = list()
    with open(filepath) as f:
        for line in f:
            data.append(line.split())
    row, col = int(data[0][0]), int(data[0][1])
    train_data = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            train_data[i][j] = int(data[i + 1][j])
    return [row, col], train_data


def compute_mean(data):
    mu = np.zeros((data.shape[0], 1))
    for i in range(data.shape[0]):
        mu[i] = np.mean(data[i])
    return mu


def compute_cov(data):
    cov = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        cov[i] = np.cov(data[i])
    return cov


def pca(data, k = 5):
    vec = np.dot(data, np.transpose(data))
    u, v = np.linalg.eig(vec)
    u.sort(axis=0)
    u_k_r = u[-k - 1:-1]
    u_k = u_k_r[::-1]
    indx = np.where(u == u_k[0])
    vec = v[:, indx[0]]
    v_k = vec

    for i in range(1, k):
        indx = np.where(u == u_k[i])
        vec = v[:, indx[0]]
        v_k = np.append(v_k, vec, axis=1)
    return v_k.real

def plot(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """

    V_plt = np.append(V.reshape(-1), V[0, :]).reshape((-1, 2))
    ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
    ax.scatter(V[:, 0], V[:, 1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def task_2():
    size, train_data = read_train_data('data/hands_aligned_train.txt.new')
    mu = compute_mean(train_data)
    v_k = pca(train_data)
    phi = v_k
    shape = np.zeros((train_data.shape[0], 1))
    h = [-0.4, -0.2, 0.0, 0.2, 0.4]
    var = np.dot(phi, h)
    for i in range(train_data.shape[0]):
        shape[i] = mu[i] + var[i]

# Visualization
    w = np.zeros((int(shape.shape[0]/2), 2))
    for i in range(w.shape[0]):
        w[i][0] = shape[i]
        w[i][1] = shape[56 + i]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.clear()
    plot(ax, w)
    plt.show()


def task_3():
    pass


task_2()

task_3()