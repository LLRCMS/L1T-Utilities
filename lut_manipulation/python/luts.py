import math as m
import numpy as np
from rootpy.plotting import Hist2D, Hist3D




def th3_to_txt(histo, output_file, output_bits, version, bin_order=[0,1,2],xname='x', yname='y', zname='z'):
    nbins = histo.nbins(0)*histo.nbins(1)*histo.nbins(2)
    address_bits = int(m.log(nbins)*m.log(m.exp(1))/m.log(2))
    header = '#<header> {0} {1} {2} </header>'.format(version,address_bits, output_bits)
    address = 0
    with open(output_file, 'w') as output:
        print >>output, header
        for bx in histo.bins_range(bin_order[0]):
            bs = [0,0,0]
            bs[bin_order[0]] = bx
            for by in histo.bins_range(bin_order[1]):
                bs[bin_order[1]] = by
                for bz in histo.bins_range(bin_order[2]):
                    bs[bin_order[2]] = bz
                    value = histo[bs[0],bs[1],bs[2]].value
                    if value>=2**output_bits: value = 2**output_bits-1
                    print >>output, address, int(value), '# {0}={1},{2}={3},{4}={5}'.format(xname,bx-1, yname,by-1, zname,bz-1)
                    address += 1


# Only for 3D histograms for the moment
# extrapolate linearly along specified axis
def extrapolate_1d(histo, axis):
    for b1 in histo.bins_range((axis+1)%3):
        bins = [0,0,0]
        bins[(axis+1)%3] = b1
        for b2 in histo.bins_range((axis+2)%3):
            bins[(axis+2)%3] = b2
            x = []
            y = []
            maxi = -1
            # compute extrapolation using a linear fit
            for b3 in histo.bins_range(axis):
                bins[axis] = b3
                center = histo.axis(axis).GetBinCenter(b3)
                value = histo[tuple(bins)].value
                if value>0 and value<256:
                    x.append(center)
                    y.append(value)
                if value>maxi:
                    maxi = value
            if len(x)>0:
                fit = np.poly1d(np.polyfit(x,y,1))
            # apply extrapolation
            for b3 in histo.bins_range(axis):
                bins[axis] = b3
                center = histo.axis(axis).GetBinCenter(b3)
                if len(x)>=2 and center>max(x):
                    histo[tuple(bins)].value = fit(center)
                elif len(x)==1 and center>max(x):
                    histo[tuple(bins)].value = y[0]
                elif len(x)==0:
                    histo[tuple(bins)].value = 9999.

#def fill_holes(histo):
    #for beta in histo.bins_range(0):
        #for bet in histo.bins_range(2):
            #values = []
            ## compute extrapolation
            #for bntt in histo.bins_range(1):
                #values.append(histo[beta,bntt,bet].value)
