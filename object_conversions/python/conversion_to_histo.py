import numpy as np
from rootpy.plotting import Hist2D

# 'function' must be able to take an array of (2D) array as input
def function2th2(function, binsx, binsy, titlex='', titley=''):
    histo = Hist2D(*(binsx+binsy))
    histo.SetXTitle(titlex)
    histo.SetYTitle(titley)
    # Prepare array of inputs, one entry for each bin
    values = []
    for bx in histo.bins_range(0):
        x = histo.GetXaxis().GetBinCenter(bx)
        for by in histo.bins_range(1):
            y = histo.GetYaxis().GetBinCenter(by)
            values.append([x,y])
    # Call function for each value
    results = function(np.array(values))
    for result,value in zip(results, values):
        bx = histo.GetXaxis().FindBin(value[0])
        by = histo.GetYaxis().FindBin(value[1])
        histo[bx,by].value = result 
    return histo
