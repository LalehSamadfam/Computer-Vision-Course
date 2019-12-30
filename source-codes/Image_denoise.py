import cv2
import numpy as np
import maxflow


def index_convertor(i, j, col):
    return i * (col - 1) + j + i


def is_node(i, j, row, col):
    if (i >= 0 and i < row) and (j >= 0 and j < col):
        return True
    return False


def question_3(I, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2):
    row, col = I.shape[0], I.shape[1]

    ### 1) Define Graph
    g = maxflow.Graph[float]()

    ### 2) Add pixels as nodes
    nodes = g.add_nodes(I.size)

    ### 3) Compute Unary cost
    unary_cost = np.zeros((row, col, 2))
    for i in range(unary_cost.shape[0]):
        for j in range(unary_cost.shape[1]):
            x = I[i][j]
            unary_cost[i][j][0] = -1 * np.log((rho ** x) * (1 - rho) ** (1 - x))
            unary_cost[i][j][1] = -1 * np.log(((1 - rho) ** x) * (rho ** (1 - x)))

    ### 4) Add terminal edges
    for i in range(row):
        for j in range(col):
            if i == 0 and j == 0:
                g.add_tedge(nodes[index_convertor(i, j, col)], unary_cost[i][j][0], unary_cost[i][j][1] +
                            pairwise_cost_same)
            elif i == row - 1 and j == col - 1:
                g.add_tedge(nodes[index_convertor(i, j, col)], unary_cost[i][j][0] + pairwise_cost_same,
                            unary_cost[i][j][1])
            else:
                g.add_tedge(nodes[index_convertor(i, j, col)], unary_cost[i][j][0] + pairwise_cost_same,
                            unary_cost[i][j][1] + pairwise_cost_same)

    ### 5) Add Node edges
    ### Vertical Edges
    for i in range(row):
        for j in range(col):
            if is_node(i, j + 1, row, col):
                right_edge = index_convertor(i, j + 1, col)
                g.add_edge(nodes[index_convertor(i, j, col)], nodes[right_edge], pairwise_cost_diff - 2 *
                           pairwise_cost_same, pairwise_cost_diff)

    ### Horizontal edges
    # (Keep in mind the structure of neighbourhood and set the weights according to the pairwise potential)
    for i in range(row):
        for j in range(col):
            if is_node(i + 1, j, row, col):
                bottom_edge = index_convertor(i + 1, j, col)
                g.add_edge(nodes[index_convertor(i, j, col)], nodes[bottom_edge], pairwise_cost_diff - 2 *
                           pairwise_cost_same, pairwise_cost_diff)

    ### 6) Maxflow
    flow = g.maxflow()

    Denoised_I = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            Denoised_I[i][j] = g.get_segment(nodes[index_convertor(i, j, col)])

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()
    return


def delta(wn, wm):
    if wn == wm:
        return 1
    return 0


def question_4(I, rho=0.6):
    pairwise_cost_diff = 10
    labels = np.unique(I).tolist()

    I_copy = I.copy()
    row, col = I.shape[0], I.shape[1]

    Denoised_I = I.copy()
    prev_Denoise = np.zeros_like(Denoised_I)
    ### Use Alpha expansion binary image for each label
    converge = False
    iter = 0
    while not converge:

        if np.sum(np.abs(np.subtract(Denoised_I, prev_Denoise))) == 0:
            converge = True

        prev_Denoise = Denoised_I
        iter += 1

        for label in labels:

            ### 1) Define Graph
            g = maxflow.Graph[float]()

            ### 2) Add pixels as nodes
            nodes = g.add_nodes(I.size)

            ### 3) Compute Unary cost
            unary_cost = np.zeros((row, col, 2))
            for i in range(row):
                for j in range(col):
                    x = I_copy[i][j]
                    if x == label:
                        unary_cost[i][j][1] = rho
                    else:
                        unary_cost[i][j][0] = (1 - rho) / 2

            for i in range(row):
                for j in range(col):
                    x = I_copy[i][j]
                    ### 4) Add terminal edges
                    if x == label:
                        g.add_tedge(nodes[index_convertor(i, j, col)], unary_cost[i][j][0], np.Inf)
                    else:
                        g.add_tedge(nodes[index_convertor(i, j, col)], unary_cost[i][j][0], unary_cost[i][j][1])

            for i in range(row):
                for j in range(col):
                    ### 5) Add Node edges
                    ### Vertical Edges
                    if is_node(i, j + 1, row, col):
                        right_edge = index_convertor(i, j + 1, col)

                        # if both are alpha
                        if I_copy[i, j] == label and I_copy[i, j + 1] == label:
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[right_edge], 0, 0)

                        # if first is alpha and second is beta
                        if I_copy[i, j] == label and I_copy[i, j + 1] != label:
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[right_edge], pairwise_cost_diff, 0)

                        # if first is beta and second is alpha
                        elif I_copy[i, j] != label and I_copy[i, j + 1] == label:
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[right_edge], 0, pairwise_cost_diff)

                        # if both are beta
                        elif I_copy[i, j] != label and I_copy[i, j + 1] != label and I_copy[i, j] == I_copy[i, j + 1]:
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[right_edge],
                                       pairwise_cost_diff, pairwise_cost_diff)


                        # if one is beta and the other is gamma
                        elif I_copy[i, j] != label and I_copy[i, j + 1] != label and I_copy[i, j] != I_copy[i, j + 1]:

                            # add node k
                            g.add_nodes(1)

                            # k and right edge
                            g.add_edge(nodes[index_convertor(i, -1, col)], nodes[right_edge],
                                       np.Inf, pairwise_cost_diff)

                            # k and left edge
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[index_convertor(i, -1, col)],
                                       pairwise_cost_diff, np.Inf)

                            # add tedge for k
                            g.add_tedge(nodes[index_convertor(i, -1, col)], 0,
                                        pairwise_cost_diff)

            for i in range(row):
                for j in range(col):
                    ### Horizontal edges
                    # (Keep in mind the structure of neighbourhood and set the weights according to the pairwise potential)
                    if is_node(i + 1, j, row, col):
                        bottom_edge = index_convertor(i + 1, j, col)
                        # if both are alpha
                        if I_copy[i, j] == label and I_copy[i + 1, j] == label:
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[bottom_edge], 0, 0)

                        # if first is alpha and second is beta
                        elif I_copy[i, j] == label and I_copy[i + 1, j] != label:
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[bottom_edge],
                                       pairwise_cost_diff, 0)

                        # if first is beta and second is alpha
                        elif I_copy[i, j] != label and I_copy[i + 1, j] == label:
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[bottom_edge],
                                       0, pairwise_cost_diff)

                        # if both are beta
                        elif I_copy[i, j] != label and I_copy[i + 1, j] != label and I_copy[i, j] == I_copy[i + 1, j]:
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[bottom_edge],
                                       pairwise_cost_diff, pairwise_cost_diff)

                        # if one is beta and the other is gamma
                        elif I_copy[i, j] != label and I_copy[i + 1, j] != label and I_copy[i, j] != I_copy[i + 1, j]:

                            # add node k
                            g.add_nodes(1)

                            # k and right edge
                            g.add_edge(nodes[index_convertor(i, -1, col)], nodes[bottom_edge],
                                       np.Inf, pairwise_cost_diff)

                            # k and left edge
                            g.add_edge(nodes[index_convertor(i, j, col)], nodes[index_convertor(i, -1, col)],
                                       pairwise_cost_diff, np.Inf)

                            # add tedge for k
                            g.add_tedge(nodes[index_convertor(i, -1, col)], 0,
                                        pairwise_cost_diff)

            ### 6) Maxflow
            flow = g.maxflow()

            for i in range(row):
                for j in range(col):
                    if (g.get_segment(nodes[index_convertor(i, j, col)])) == 0:
                        Denoised_I[i][j] = label

            I_copy = Denoised_I.copy()

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return


def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.35)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.55)

    ### Call solution for question 4
    question_4(image_q4, rho=0.8)
    return


if __name__ == "__main__":
    main()
