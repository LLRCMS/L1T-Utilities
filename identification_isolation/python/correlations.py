import numpy as np

from sklearn import linear_model
from sklearn.externals import joblib
from sklearn import cross_validation

from rootpy.plotting import Hist2D
from rootpy.io import root_open
from root_numpy import root2array


def fit_linear(filename, treename, inputname, targetname, test=False):
    # Reading inputs and targets
    branches = [inputname, targetname]
    data = root2array(filename, treename=treename, branches=branches)
    data = data.view((np.float64, len(data.dtype.names)))
    # Extract and format inputs and targets from numpy array
    inputs = data[:, [0]].astype(np.float32)
    targets = data[:, [1]].astype(np.float32).ravel()
    # if test requested, use 60% of events for training and 40% for testing
    inputs_train = inputs
    targets_train = targets
    if test:
        inputs_train, inputs_test, targets_train, targets_test = cross_validation.train_test_split(inputs, targets, test_size=0.4, random_state=0)
    regressor = linear_model.LinearRegression()
    regressor.fit(inputs_train, targets_train)
    if test:
        score_test = regressor.score(inputs_test, targets_test)
        score_train = regressor.score(inputs_train, targets_train)
        print '    Train score =', score_train
        print '    Test score =', score_test
        print regressor.coef_
        print regressor.intercept_
    return regressor


def main(inputfile, tree, input, target, outputfile, name, test=False):
    regressor = fit_linear(filename=inputfile, treename=tree, inputname=input, targetname=target, test=test)
    #if os.path.splitext(outputfile)[1]!='.root': outputfile += '.root'
    #with root_open(outputfile, 'recreate') as output_file:
        #store(regressor=regressor, name=name, inputs=inputs, outputfile=output_file)
    return regressor

if __name__=='__main__':
    import optparse
    usage = 'usage: python %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--inputfile', dest='input_file', help='Input file', default='tree.root')
    parser.add_option('--tree', dest='tree_name', help='Tree in the input file', default='tree')
    parser.add_option('--input', dest='input', help='Input variable', default='x')
    parser.add_option('--target', dest='target', help='Target variable', default='target')
    parser.add_option('--outputfile', dest='output_file', help='Output file', default='results.root')
    parser.add_option('--name', dest='name', help='Name used to store the regression results in the output file', default='regression')
    parser.add_option('--test', action="store_true", dest='test', help='Flag to test regression on a test sample', default=False)
    (opt, args) = parser.parse_args()
    main(inputfile=opt.input_file, tree=opt.tree_name, input=opt.input, target=opt.target, outputfile=opt.output_file, name=opt.name, test=opt.test)
