
import copy
import numpy as np

from rootpy.plotting import Hist, Hist2D, Graph
from rootpy.io import root_open
from root_numpy import fill_hist, root2array


def efficiency_graph(pass_function, function_inputs, xs, bins=None, error=0.005):
    pass_results = pass_function(function_inputs)
    if not bins: # Automatic binning
        # Compute the number of bins such that the error on the efficiency is equal to 'error' in each bin
        # The calculation is based on binomial errors and assumes that the efficiency is flat (that the distributions of all and selected events are the same)
        k = float(np.count_nonzero(pass_results))
        n = float(len(pass_results))
        percentiles = [0.,100.]
        if k>0: 
            nbins = (error*n)**2/k / (1-k/n)
            # Compute the bin boundaries with the same number of events in all bins
            percentiles = np.arange(0., 100., 100./nbins)
            percentiles[-1] = 100.
        bins = np.unique(np.percentile(xs, percentiles))
    # Fill histograms of selected and all events and compute efficiency
    histo_pass = Hist(bins)
    histo_total = Hist(bins)
    fill_hist(histo_pass, xs, pass_results)
    fill_hist(histo_total, xs)
    efficiency = Graph()
    efficiency.Divide(histo_pass, histo_total)
    return efficiency

