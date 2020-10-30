import numpy as np
import sys

sys.path.extend(['../'])
from graph import tools
import networkx as nx

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

# Edge format: (origin, neighbor)

def joint_info(num_node):
    self_link_1 = [(i, i) for i in range(num_node[0])]
    inward_1 = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14)]
    outward_1 = [(j, i) for (i, j) in inward_1]
    neighbor_1 = inward_1 + outward_1

    self_link_2 = [(i, i) for i in range(num_node[1])]
    inward_2 = [(1, 0), (2, 0), (4, 3), (5, 3), (7, 6), (8, 6), (6, 3), (3, 0),
              ]
    outward_2 = [(j, i) for (i, j) in inward_2]
    neighbor_2 = inward_2 + outward_2

    self_link_3 = [(i, i) for i in range(num_node[2])]
    inward_3 = [(1, 0), (2, 1), (3, 1), (4, 1), (5, 1)]
    outward_3 = [(j, i) for (i, j) in inward_3]
    neighbor_3 = inward_3 + outward_3

    self_link = [self_link_1, self_link_2, self_link_3]
    inward = [inward_1, inward_2, inward_3]
    outward = [outward_1, outward_2, outward_3]
    neighbor = [neighbor_1,neighbor_2,neighbor_3]
    return self_link, inward, outward,neighbor

def partion():
    partion1 = [(0, 0), (0, 1), (1, 14), (1, 16), (2, 15),
               (2, 17), (3, 2), (3, 5), (4, 3), (4, 4), (5, 6),
               (5, 7), (6, 8), (6, 11), (7, 9), (7,10), (8, 12),
               (8, 13)]
    partion2 = [(0, 0), (0, 1), (0, 2), (1, 3), (1, 6),
               (2, 4), (3, 5), (4, 7), (5, 8)]

    joint_to_part = [partion1,partion2]
    return joint_to_part

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = [18, 9, 6]
        self.self_link, self.inward, self.outward,self.neighbor = joint_info(self.num_node)
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
            M1 = tools.get_partion_matrix(self.num_node[i], self.num_node[i + 1],self.partion[i])
            M.append(M1)
        return M


if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')
