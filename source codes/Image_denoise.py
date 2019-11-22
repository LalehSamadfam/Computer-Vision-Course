import cv2
import numpy as np
import maxflow

def index_convertor(i, j, col):
    return i * (col - 1) + j + i


def is_node(i, j, row, col):
    if (i >= 0 and i < row) and (j >= 0 and j < col):
        return True
    return False


def question_3(I,rho=0.7,pairwise_cost_same=0.005,pairwise_cost_diff=0.2):
    row, col = I.shape[0], I.shape[1]

    ### 1) Define Graph
    g = maxflow.Graph[float]()

    ### 2) Add pixels as nodes
    nodes = g.add_nodes(I.size)
    print(nodes)

    ### 3) Compute Unary cost
    unary_cost = np.zeros((row, col, 2)) #TODO
    for i in range(unary_cost.shape[0]):
        for j in range(unary_cost.shape[1]):
            x = I[i][j]
            unary_cost[i][j][0] = -1 * np.log((rho ** x) * (1 - rho) ** (1 - x))
            unary_cost[i][j][1] = -1 * np.log(((1 - rho) ** x) * (rho ** (1 - x)))
            #print('x, u0, u1', x, unary_cost[i][j][0], unary_cost[i][j][1])

    ### 4) Add terminal edges
    for i in range(row):
        for j in range(col):
            if i == 0 and j == 0:
                g.add_tedge(nodes[index_convertor(i, j, col)], unary_cost[i][j][0], unary_cost[i][j][1] +
                            pairwise_cost_same)  #check if col is compatible #TODO
            elif i == row - 1 and j == col - 1:
                g.add_tedge(nodes[index_convertor(i, j, col)], unary_cost[i][j][0] + pairwise_cost_same,
                            unary_cost[i][j][1])
            else:
                g.add_tedge(nodes[index_convertor(i, j, col)], unary_cost[i][j][0] + pairwise_cost_same,
                            unary_cost[i][j][1] + pairwise_cost_same)

            #print(g.g)

    ### 5) Add Node edges
    ### Vertical Edges
    for i in range(row):
        for j in range(col):
            if is_node(i, j + 1, row, col):
                right_edge = index_convertor(i, j + 1, col)
                g.add_edge(nodes[index_convertor(i, j, col)], nodes[right_edge], pairwise_cost_diff - 2 * pairwise_cost_same, pairwise_cost_diff )

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)
    for i in range(row):
        for j in range(col):
            if is_node(i + 1, j, row, col):
                bottom_edge = index_convertor(i + 1, j, col)
                g.add_edge(nodes[index_convertor(i, j, col)], nodes[bottom_edge], pairwise_cost_diff - 2 * pairwise_cost_same,
                           pairwise_cost_diff)

    ### 6) Maxflow
    g.maxflow()
    Denoised_I = np.zeros_like(I)

    for i in range(row):
        for j in range(col):
            Denoised_I[i][j] = g.get_segment(nodes[index_convertor(i, j, col)])


    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I,rho=0.6):

    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    ### Use Alpha expansion binary image for each label

    ### 1) Define Graph

    ### 2) Add pixels as nodes

    ### 3) Compute Unary cost

    ### 4) Add terminal edges

    ### 5) Add Node edges
    ### Vertical Edges

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow


    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2)
    #question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.35)
    #question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.55)

    ### Call solution for question 4
    #question_4(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()



