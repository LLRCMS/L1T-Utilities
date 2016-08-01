import numpy as np

def graph2array(graph):
    xs = np.array([graph.GetX()[p] for p in range(graph.GetN())])
    ys = np.array([graph.GetY()[p] for p in range(graph.GetN())])
    return np.column_stack((xs,ys))
