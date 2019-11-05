import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

def binary_cost(x,y):
    return None


def unary_cost(x):
    return None


# FUNCTIONS
def find_path(mat, vertices, image, fixed_vertice):
    for j in range(mat.shape[1]):
        mat[0][j] = unary_cost(x=0)  # TODO define unary and binary cost and correct the values of states (x, 100, ..)
    min_paths = np.zeros((mat.shape[0], mat.shape[1] - 1))
    for j in range(mat.shape[1]):
        for i in range(1, mat.shape[0]):
            same_state = 100  # binary_cost with same state from prev node
            prev_state = 100
            next_state = 100
            if j > 1:
                prev_state = 100
            if j < mat.shape[0]:
                next_state = 100
            mat[i][j] = np.min(same_state, prev_state, next_state) + unary_cost()
            min_paths[i][j] = 0  # TODO the state which gives the min dist
    for j in range(mat.shape[1] - 1):
        pass  # TODO track the min path
    return None


def update_cost(mat, vertices, image):
    fixed_vertice = np.random.random(vertices.shape[1])
    np.roll(vertices, 2 * fixed_vertice)
    path = find_path(mat, vertices, image, fixed_vertice)
    np.roll(path, -2 * fixed_vertice)
    return path


def update_vertices(old, new):
    pass


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 1
    states = 9

    for t in range(n_steps):
        print(V.size)
        cost_mat = np.zeros((states, V.shape[1]))
        path = update_cost(cost_mat, V, Im)
        update_vertices(V, path)
        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    #run('images/coffee.png', radius=100)
