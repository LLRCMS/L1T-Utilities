from object_conversions.conversion_to_histo import function2th2
from identification_isolation import quantile_regression, correlations

import copy
import numpy as np

# Always import ROOT related stuff after everything else
from rootpy.plotting import Hist2D
from rootpy.io import root_open


# Compound of multivariate isolation cuts and input mappings
class isolation_cuts:
    def __init__(self, iso_regression, input_mappings):
        self.iso_regression = iso_regression
        # dictionary input index -> function to be applied on inputs
        self.input_mappings = input_mappings
        for index, mapping in self.input_mappings.items():
            self.input_mappings[index] = np.vectorize(mapping)

    def predict(self, values):
        # Apply input mappings
        mapped_inputs = np.array(values, dtype=np.float64)
        for index,mapping in self.input_mappings.items():
            # Apply mapping on column 'index'
            #mapping = np.vectorize(fct)
            mapped_inputs_i = mapping(mapped_inputs[:,[index]])
            # Replace column 'index' with mapped inputs
            mapped_inputs = np.delete(mapped_inputs, index, axis=1)
            mapped_inputs = np.insert(mapped_inputs, [index], mapped_inputs_i, axis=1)
        # Apply iso regression on mapped inputs
        return self.iso_regression.predict(mapped_inputs)






def map_pu_variable(iso_regression, pu_regression, name):
    binning_eta = quantile_regression.binning['abs(ieta)']
    binning_ntt = quantile_regression.binning['ntt']
    histo = Hist2D(*(binning_eta+binning_ntt), name=name)
    histo.SetXTitle('|i#eta|')
    histo.SetYTitle('n_{TT}')
    a = pu_regression.intercept_
    b = pu_regression.coef_[0]
    eg_isolation_cuts = isolation_cuts(iso_regression=iso_regression, input_mappings={1:(lambda x:max(0.,(x-a)/b))})
    # Prepare array of inputs, one entry for each bin
    values = []
    for bx in histo.bins_range(0):
        x = histo.GetXaxis().GetBinCenter(bx)
        for by in histo.bins_range(1):
            y = histo.GetYaxis().GetBinCenter(by)
            values.append([x,y])
    # Compute isolation cuts for each value
    iso_cuts = eg_isolation_cuts.predict(values)
    for cut,value in zip(iso_cuts, values):
        bx = histo.GetXaxis().FindBin(value[0])
        by = histo.GetYaxis().FindBin(value[1])
        histo[bx,by].value = cut
    return histo



def main(inputfile, tree, outputfile, name, test=False, inputs=['abs(ieta)','rho'], target='iso'):
    eff = 0.9
    print 'Deriving rho->ntt map'
    pu_regression = correlations.fit_linear(inputfile, tree, 'rho', 'ntt', test=False)
    # Train isolation fixed working points
    print 'Deriving isolation working point'
    iso_regression = quantile_regression.main(inputfile, tree, inputs, target, outputfile, name, eff, test) 
    print 'Applying rho->ntt map'
    a = pu_regression.intercept_
    b = pu_regression.coef_[0]
    eg_isolation_cuts = isolation_cuts(iso_regression=iso_regression, input_mappings={1:(lambda x:max(0.,(x-a)/b))})
    with root_open(outputfile, 'recreate') as output_file:
        histo = function2th2(eg_isolation_cuts.predict, quantile_regression.binning['abs(ieta)'], quantile_regression.binning['ntt'])
        #histo = map_pu_variable(iso_regression, pu_regression, name)
        output_file.cd()
        histo.Write()



if __name__=='__main__':
    import optparse
    usage = 'usage: python %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--inputfile', dest='input_file', help='Input file', default='tree.root')
    parser.add_option('--tree', dest='tree_name', help='Tree in the input file', default='tree')
    parser.add_option('--outputfile', dest='output_file', help='Output file', default='results.root')
    parser.add_option('--name', dest='name', help='Name used to store the regression results in the output file', default='egamma_isolation')
    parser.add_option('--test', action="store_true", dest='test', help='Flag to test regression on a test sample', default=False)
    parser.add_option('--inputs', dest='inputs', help='List of input variables of the form "var1,var2,..."', default='abs(ieta),rho')
    parser.add_option('--target', dest='target', help='Target variable', default='iso')
    (opt, args) = parser.parse_args()
    inputs = opt.inputs.replace(' ','').split(',')
    main(inputfile=opt.input_file, tree=opt.tree_name, outputfile=opt.output_file, name=opt.name, test=opt.test, inputs=inputs, target=opt.target)
