import numpy as np
from rootpy.plotting import Hist2D, Hist3D
from root_numpy import fill_hist

# convert a ROOT TGraph (without errors) to a Numpy array
def graph2array(graph):
    xs = np.array([graph.GetX()[p] for p in range(graph.GetN())])
    ys = np.array([graph.GetY()[p] for p in range(graph.GetN())])
    return np.column_stack((xs,ys))


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



# 'function' must be able to take an array of (3D) array as input
def function2th3(function, binsx, binsy, binsz, titlex='', titley='', titlez=''):
    histo = Hist3D(*(binsx+binsy+binsz))
    histo.SetXTitle(titlex)
    histo.SetYTitle(titley)
    histo.SetZTitle(titlez)
    # Prepare array of inputs, one entry for each bin
    values = []
    for bx in histo.bins_range(0):
        x = histo.GetXaxis().GetBinCenter(bx)
        for by in histo.bins_range(1):
            y = histo.GetYaxis().GetBinCenter(by)
            for bz in histo.bins_range(2):
                z = histo.GetZaxis().GetBinCenter(bz)
                values.append([x,y,z])
    # Call function for each value
    results = function(np.array(values))
    for result,value in zip(results, values):
        bx = histo.GetXaxis().FindBin(value[0])
        by = histo.GetYaxis().FindBin(value[1])
        bz = histo.GetZaxis().FindBin(value[2])
        histo[bx,by,bz].value = result 
    return histo


def events2th3(inputs, values, binsx, binsy, binsz, titlex='', titley='', titlez=''):
    # Histo used to fill values
    histo = Hist3D(*(binsx+binsy+binsz))
    # Histo used to count entries
    histo_count = Hist3D(*(binsx+binsy+binsz))
    histo.SetXTitle(titlex)
    histo.SetYTitle(titley)
    histo.SetZTitle(titlez)
    fill_hist(histo, inputs, values)
    fill_hist(histo_count, inputs)
    # Compute mean values in bins
    histo.Divide(histo_count)
    return histo
