import copy
import numpy as np
from identification_isolation.efficiency import efficiency_graph

from root_numpy import root2array



def test_efficiency(functions, function_inputs, variables, inputfile, tree, selection):
    # Retrieve data from tree
    ninputs = len(function_inputs)
    branches = copy.deepcopy(function_inputs)
    branches.extend(variables)
    data = root2array(inputfile, treename=tree, branches=branches, selection=selection)
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    # Compute efficiencies along each variable and for each function
    graphs = []
    try:
        for ifu, function in enumerate(functions):
            for i,variable in enumerate(variables):
                xs  = data[:, [ninputs+i]].astype(np.float32).ravel()
                graphs.append(efficiency_graph(pass_function=function,\
                                               function_inputs=inputs,\
                                               xs=xs))
                graphs[-1].SetName('efficiency_{}_{}'.format(ifu,variable))
    except TypeError:
        for i,variable in enumerate(variables):
            xs  = data[:, [ninputs+i]].astype(np.float32).ravel()
            graphs.append(efficiency_graph(pass_function=functions,\
                                           function_inputs=inputs,\
                                           xs=xs))
            graphs[-1].SetName('efficiency_'+variable)
    return graphs

