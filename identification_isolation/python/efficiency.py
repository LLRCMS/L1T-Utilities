
import copy
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from rootpy.plotting import Hist, Hist2D, Graph
from rootpy.io import root_open
from rootpy.ROOT import TEfficiency
from root_numpy import fill_hist, fill_graph, root2array




def efficiency_inclusive(pass_function, function_inputs):
    pass_results = pass_function(function_inputs)
    k = float(np.count_nonzero(pass_results))
    n = float(len(pass_results))
    efficiency = 0.
    lower = 0.
    upper = 0.
    if n>0.:
        confidence = 0.682689492137
        efficiency = k/n
        lower = TEfficiency.ClopperPearson(n,k,confidence,False)
        upper = TEfficiency.ClopperPearson(n,k,confidence,True)
    return efficiency, lower, upper


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

def efficiency_bdt(pass_function, function_inputs, xs):
    pass_results = pass_function(function_inputs)
    xs_train = xs.reshape(-1,1)
    clf = GradientBoostingClassifier()
    clf.fit(xs_train, pass_results)
    graph = Graph(100)
    xs_graph = np.linspace(np.amin(xs), np.amax(xs), num=100)
    probas = clf.predict_proba(xs_graph.reshape(-1,1))[:, [1]]
    print probas
    fill_graph(graph, np.column_stack((xs_graph, probas)))
    print np.column_stack((xs_graph, probas))
    return graph


#def cuts_efficiencies(pass_functions, function_inputs):
    #return [efficiency_inclusive(pass_function=pass_function, function_inputs=function_inputs) for pass_function in pass_functions]


#def cuts_efficiencies(working_points, pass_functions, function_inputs):
    #efficiency_curve = Graph(len(pass_functions))
    #for i,(wp,pass_function) in enumerate(zip(working_points,pass_functions)):
        #eff, low, up = efficiency_inclusive(pass_function=pass_function, function_inputs=function_inputs)
        #efficiency_curve.SetPoint(i, wp, eff)
        #efficiency_curve.SetPointError(i, 0,0, eff-low, up-eff)
    #return efficiency_curve

