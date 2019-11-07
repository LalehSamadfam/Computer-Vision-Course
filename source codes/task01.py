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

def stiffness(vertex, state, prev_state):
    return 0

def elasticity(vertices, vertex, state, prev_state, d):
    x, y = get_state_position(state)
    x_p, y_p = get_state_position(prev_state)
    cost = np.sqrt(((vertices[vertex][0] + x) - (vertices[vertex - 1][0] - x_p)) ** 2 +
                   ((vertices[vertex][1] + y) - (vertices[vertex - 1][1] - y_p)) ** 2 - d)
    return cost


def binary_cost(vertices, vertex, state, prev_state, mean_dist): # internal cost
    return stiffness(vertex, state, prev_state) + elasticity(vertices, vertex, state, prev_state, mean_dist)


def get_state_position(state):
    if state == 0:
        return -1, -1
    elif state == 1:
        return 0, -1
    elif state == 2:
        return 1, -1
    elif state == 3:
        return -1, 0
    elif state == 4:
        return 0, 0
    elif state == 5:
        return 1, 0
    elif state == 6:
        return -1, 1
    elif state == 7:
        return 0, 1
    else:
        return 1, 1


def unary_cost(state, vertex, grad_x, grad_y): # external cost
    x, y = get_state_position(state)
    #print(x,y)
    i = vertex[0] + x
    j = vertex[1] + y
    #print(i, j)
    return -1 * ((grad_x[i][j]) ** 2 + (grad_y[i][j]) ** 2)


# FUNCTIONS
def find_path(mat, vertices, image, mean_dist):
    alpha = 1000
    #grad_x, grad_y = np.gradient(image)
    sobelx64f = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    grad_y = np.uint8(abs_sobel64f)
    sobelx64f = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    grad_x = np.uint8(abs_sobel64f)
    for i in range(mat.shape[0]):
        mat[i][0] = unary_cost(i, vertices[0], grad_x, grad_y)
    print('mat:')
    print(mat)
    min_paths = np.zeros((mat.shape[0], mat.shape[1]))
    for j in range(1, mat.shape[1]):
        for i in range(mat.shape[0]):
            costs = np.zeros(vertices.shape[0])
            for k in range(costs.size):
                costs[k] = binary_cost(vertices, j, i, k, mean_dist)
            #input('new cost will be printed')
            #print('i, j=', i, j)
            #print('costs:')
            #print(costs)
            unary = unary_cost(i, vertices[j], grad_x, grad_y)
            #print('unary= ', unary)
            mat[i][j] = alpha * np.min(costs) + unary
            single_path = np.argmin(costs)
            min_paths[i][j] = single_path
    #print('mat:')
    #print(mat)
    #print(min_paths)
    path = []
    end_vertex = np.argmin(mat[:, mat.shape[1] - 1])
    #print(end_vertex)
    path.append(end_vertex)
    #print(path)
    for j in range(min_paths.shape[1] - 1):
       # print(end_vertex)
        #print(min_paths.shape[1] - j - 1)
        end_vertex = int(min_paths[end_vertex][min_paths.shape[1] - j - 1])
        path.append(end_vertex)
    #print(path)
    return path


def update_cost(mat, vertices, image, mean_dist):
    fixed_vertex = np.random.randint(vertices.shape[0])
    rolled = np.roll(vertices, 2 * fixed_vertex)
    print(rolled)
    rolled_path = find_path(mat, rolled, image, mean_dist)
    path = np.roll(rolled_path, -2 * fixed_vertex)
    return path


def update_vertices(old, new_states):
    for i in range(old.shape[0]):
        x, y = get_state_position(new_states[i])
        old[i][0] = old[i][0] + x
        old[i][1] = old[i][1] + y
    return old

def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200
    states = 9
    mean_dist = np.sqrt((V[0][0] - V[1][0]) ** 2 + (V[0][1] - V[1][1]) ** 2)

    for t in range(n_steps):
        cost_mat = np.zeros((states, V.shape[0]))
        path = update_cost(cost_mat, V, Im, mean_dist)
        V = update_vertices(V, path)
        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    #run('images/coffee.png', radius=100)
