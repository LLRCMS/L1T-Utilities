import copy

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation

from rootpy.plotting import Hist2D
from rootpy.io import root_open
from root_numpy import root2array



# Predefined binning to store regression results
binning = {}
binning['abs(ieta)'] = (30, 0.5, 30.5)
binning['rho'] = (500, 0., 50)

def fit(filename, treename, inputsname, targetname, workingpoint=0.9, test=False):
    # Reading inputs and targets
    ninputs = len(inputsname)
    branches = copy.deepcopy(inputsname)
    branches.append(targetname)
    data = root2array(filename, treename=treename, branches=branches)
    data = data.view((np.float64, len(data.dtype.names)))
    # Extract and format inputs and targets from numpy array
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets = data[:, [ninputs]].astype(np.float32).ravel()
    # if test requested, use 60% of events for training and 40% for testing
    inputs_train = inputs
    targets_train = targets
    if test:
        inputs_train, inputs_test, targets_train, targets_test = cross_validation.train_test_split(inputs, targets, test_size=0.4, random_state=0)
    # Define and fit quantile regression (quantile = workingpoint)
    # Default training parameters are used
    regressor = GradientBoostingRegressor(loss='quantile', alpha=workingpoint)
    regressor.fit(inputs_train, targets_train)
    if test:
        # Compare regression prediction with the true value and count the fraction of time it falls below
        # This should give the working point value
        predict_test = regressor.predict(inputs_test)
        compare = np.less(targets_test, predict_test)
        print 'Testing regression with inputs', inputsname, 'and working point', workingpoint
        print '    Test efficiency =', float(list(compare).count(True))/float(len(compare))
        # TODO: add 1D efficiency graphs vs input variables
    return regressor

def store(regressor, name, inputs, outputfile):
    if len(inputs)!=2:
        raise StandardError('Can only store regression result into a 2D histogram for the moment')
    histo = Hist2D(*(binning[inputs[0]]+binning[inputs[1]]), name=name)
    histo.SetXTitle(inputs[0])
    histo.SetYTitle(inputs[1])
    for bx in histo.bins_range(0):
        x = histo.GetXaxis().GetBinCenter(bx)
        for by in histo.bins_range(1):
            y = histo.GetYaxis().GetBinCenter(by)
            histo[bx,by].value = regressor.predict([[x,y]])
    outputfile.cd()
    histo.Write()


def main():
    import optparse
    usage = 'usage: python %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--inputfile', dest='input_file', help='Input file', default='tree.root')
    parser.add_option('--tree', dest='tree_name', help='Tree in the input file', default='tree')
    parser.add_option('--inputs', dest='inputs', help='List of input variables of the form "var1,var2,..."', default='x,y')
    parser.add_option('--target', dest='target', help='Target variable', default='target')
    parser.add_option('--eff', dest='eff', help='Efficiency working point', type='float', default=0.9)
    parser.add_option('--outputfile', dest='output_file', help='Output file', default='results.root')
    parser.add_option('--name', dest='name', help='Name used to store the regression results in the output file', default='regression')
    parser.add_option('--test', action="store_true", dest='test', help='Flag to test regression on a test sample', default=False)
    (opt, args) = parser.parse_args()
    #input_file = '/data_CMS/cms/sauvan/L1/2016/IsolationValidation/ZElectron/v_3_2016-06-23/tagAndProbe_isolationValidation_2016B_ZElectron.root'
    #tree_name = 'ntTagAndProbe_IsolationValidation_Stage2_Rebuilt_tree'
    #inputs = ['abs(ieta)', 'rho']
    #target = 'iso'
    inputs = opt.inputs.replace(' ','').split(',')
    regressor = fit(filename=opt.input_file, treename=opt.tree_name, inputsname=inputs, targetname=opt.target, workingpoint=opt.eff, test=opt.test)
    with root_open(opt.output_file, 'recreate') as output_file:
        store(regressor=regressor, name=opt.name, inputs=inputs, outputfile=output_file)


if __name__=='__main__':
    main()









