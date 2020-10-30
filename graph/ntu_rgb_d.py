import sys

sys.path.extend(['../'])
from graph import tools




def joint_info(num_node):
    self_link_1 = [(i, i) for i in range(num_node[0])]
    inward_1 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
    inward_1 = [(i - 1, j - 1) for (i, j) in inward_1]
    outward_1 = [(j, i) for (i, j) in inward_1]

    self_link_2 = [(i, i) for i in range(num_node[1])]
    inward_2 = [(2, 1), (3, 1), (4, 3), (5, 1), (6, 5), (7, 1), (8, 7), (9, 1), (10, 9)]
    inward_2 = [(i - 1, j - 1) for (i, j) in inward_2]
    outward_2 = [(j, i) for (i, j) in inward_2]

    self_link_3 = [(i, i) for i in range(num_node[2])]
    inward_3 = [(2, 1), (3, 1), (4, 1), (5, 1)]
    inward_3 = [(i - 1, j - 1) for (i, j) in inward_3]
    outward_3 = [(j, i) for (i, j) in inward_3]


    self_link = [self_link_1, self_link_2, self_link_3]
    inward = [inward_1, inward_2, inward_3]
    outward = [outward_1, outward_2, outward_3]

    return self_link, inward, outward


def partion():
    partion1 = [(1, 1), (1, 2), (2, 3), (2, 4), (1, 21),
               (3, 5), (3, 6), (3, 7), (4, 8), (4, 22), (4, 23),
               (5, 9), (5, 10), (5, 11), (6, 12), (6,24), (6, 25),
               (7, 13), (7, 14), (8, 15), (8, 16),
               (9, 17), (9, 18), (10, 19), (10, 20)]
    partion1 = [(i - 1,j - 1) for (i,j) in partion1]

    partion2 = [(1, 1), (1, 2), (2, 3), (2, 4), (3, 5), (3, 6), (4, 7), (4, 8), (5, 9), (5, 10)]
    partion2 =  [(i - 1,j - 1) for (i,j) in partion2]


    joint_to_part = [partion1,partion2]

    return joint_to_part

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = [25, 10, 5]
        self.self_link, self.inward, self.outward = joint_info(self.num_node)
        self.partion = partion()
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.M = self.get_partion_metrix()


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A=[]
            for i in range(len(self.num_node)):
                A1 = tools.get_spatial_graph(self.num_node[i], self.self_link[i], self.inward[i], self.outward[i])
                A.append(A1)
        else:
            raise ValueError()
        return A

    def get_partion_metrix(self):
        M = []
        for i in range(len(self.partion)):
            M1 = tools.get_partion_matrix(self.num_node[i],self.num_node[i+1],self.partion[i])
            M.append(M1)
        return  M


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
    # M = Graph('spatial').get_partion_metrix()
    # for i in M:
    #     plt.imshow(i, cmap='gray')
    #     plt.show()
    # print(M)