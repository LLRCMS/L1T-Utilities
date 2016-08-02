import numpy as np
from utilities.numpy_utilities import find_closest
from root_numpy import hist2array, evaluate, array2hist


# Compound of multivariate regression and input mappings
class RegressionWithInputMapping:
    def __init__(self, iso_regression, input_mappings, name='isolation'):
        self.name = name
        self.iso_regression = iso_regression
        # dictionary input index -> function to be applied on inputs
        self.input_mappings = input_mappings
        # Vectorize the functions such that they can take arrays as input
        for index, mapping in self.input_mappings.items():
            self.input_mappings[index] = np.vectorize(mapping)

    def predict(self, values):
        #print 'In IsolationCuts.predict()'
        # Apply input mappings
        mapped_inputs = np.array(values, dtype=np.float64)
        for index,mapping in self.input_mappings.items():
            # Apply mapping on column 'index'
            mapped_inputs_i = mapping(mapped_inputs[:,[index]])
            # Replace column 'index' with mapped inputs
            mapped_inputs = np.delete(mapped_inputs, index, axis=1)
            mapped_inputs = np.insert(mapped_inputs, [index], mapped_inputs_i, axis=1)
        # Apply iso regression on mapped inputs
        output = self.iso_regression.predict(mapped_inputs)
        #print 'Out IsolationCuts.predict()'
        return output


class CombinedWorkingPoints:
    # TODO: Improve performance
    def __init__(self, working_points, functions, efficiency_map):
        efficiency_array = hist2array(efficiency_map)
        working_points_indices = find_closest(working_points, efficiency_array)
        self.function_index_map = efficiency_map.empty_clone()
        array2hist(working_points_indices, self.function_index_map)
        self.indices = working_points_indices
        self.functions = functions
        self.dim = len(efficiency_array.shape)

    def value(self, inputs, map_positions):
        # remove overflows (overwrite with a value just below the histogram boundary)
        upper_bounds = [self.function_index_map.bounds(axis)[1]-1e-3 for axis in range(len(self.function_index_map.axes))]
        map_positions_no_overflow = np.apply_along_axis(lambda x:np.minimum(x,upper_bounds), 1, map_positions)
        # evaluate of a 1D histograms take flatten array as input
        if self.dim==1: map_positions_no_overflow = map_positions_no_overflow.ravel()
        indices = evaluate(self.function_index_map, map_positions_no_overflow).astype(np.int32)
        # Compute isolation for all used working points
        outputs = []
        for i,function in enumerate(self.functions):
            if i in self.indices: outputs.append(function(inputs))
            else: outputs.append(np.array([]))
        #output = [self.functions[index]([input]) for index,input in zip(indices,inputs)]
        # Associate the correct working point for each entry
        output = np.zeros(len(indices))
        for i,index in enumerate(indices):
            output[i] = outputs[index][i]
        return output


