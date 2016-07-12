from object_conversions.conversion_to_histo import function2th2
from batch import batch_launcher
from identification_isolation import quantile_regression, correlations, efficiency

import copy
import os

import numpy as np
from sklearn.externals import joblib

from rootpy.plotting import Hist2D, Graph
from rootpy.io import root_open
from root_numpy import root2array, fill_graph


# Compound of multivariate isolation cuts and input mappings
class isolation_cuts:
    def __init__(self, iso_regression, input_mappings, name='isolation'):
        self.name = name
        self.iso_regression = iso_regression
        # dictionary input index -> function to be applied on inputs
        self.input_mappings = input_mappings
        # Vectorize the functions such that they can take arrays as input
        for index, mapping in self.input_mappings.items():
            self.input_mappings[index] = np.vectorize(mapping)

    def predict(self, values):
        # Apply input mappings
        mapped_inputs = np.array(values, dtype=np.float64)
        for index,mapping in self.input_mappings.items():
            # Apply mapping on column 'index'
            mapped_inputs_i = mapping(mapped_inputs[:,[index]])
            # Replace column 'index' with mapped inputs
            mapped_inputs = np.delete(mapped_inputs, index, axis=1)
            mapped_inputs = np.insert(mapped_inputs, [index], mapped_inputs_i, axis=1)
        # Apply iso regression on mapped inputs
        return self.iso_regression.predict(mapped_inputs)


def iso_parameters(inputfile, tree, name, inputs, target, effs, test):
    parameters = []
    for eff in effs:
        parameters.append({})
        parameters[-1]['inputfile'] = inputfile
        parameters[-1]['tree'] = tree
        parameters[-1]['inputs'] = '"'+','.join(inputs)+'"'
        parameters[-1]['target'] = target
        parameters[-1]['outputfile'] = name+'_{}.root'.format(eff*100)
        parameters[-1]['name'] = name
        parameters[-1]['eff'] = str(eff)
        if test: parameters[-1]['test'] = ''
    return parameters


def train_isolation_workingpoints(effs, inputfile, tree, outputdir, version, name, test=False, inputs=['abs(ieta)','ntt'], target='iso', pileupref='rho'):
    print '> Deriving {0}->{1} map'.format(pileupref, inputs[1])
    pu_regression = correlations.fit_linear(inputfile, tree, pileupref, inputs[1], test=False)
    print '> Deriving isolation working points'
    workingdir = outputdir+'/'+version
    # Replacing L1 pile-up variable (ntt) with the reference pile-up variable (rho)
    regression_inputs = copy.deepcopy(inputs)
    regression_inputs[1] = pileupref
    parameters = iso_parameters(inputfile, tree, name, regression_inputs, target, effs, test)
    batch_launcher.main(workingdir=outputdir,
                        exe='python {}/identification_isolation/python/quantile_regression.py'.format(os.environ['L1TSTUDIES_BASE']),
                        pars=parameters)
    print '> Waiting batch jobs to finish...'
    batch_launcher.wait_jobs(workingdir, wait=5)
    print '  ... Batch jobs done'
    print '> Applying {0}->{1} map'.format(pileupref, inputs[1])
    eg_isolations = []
    for i,pars in enumerate(parameters):
        # Load saved isolation regression
        eff = float(pars['eff'])
        result_dir = workingdir+'/'+name+'_{}'.format(eff*100)
        iso_regression = joblib.load(result_dir+'/'+name+'.pkl')
        # Apply rho->ntt linear mapping on isolation regression
        a = pu_regression.intercept_
        b = pu_regression.coef_[0]
        eg_isolation_cuts = isolation_cuts(name='eg_iso_'+pars['eff'], iso_regression=iso_regression, input_mappings={1:(lambda x:max(0.,(x-a)/b))})
        eg_isolations.append(eg_isolation_cuts)
    return eg_isolations



def test_isolation_workingpoints(effs, isolations, inputfile, tree, inputnames=['abs(ieta)','ntt'], targetname='iso', variables=['offl_eta','offl_pt', 'rho', 'npv']):
    # Retrieve data from tree
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    branches.extend(variables)
    data = root2array(inputfile, treename=tree, branches=branches)
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    graphs = []
    # Compute efficiencies for each working point and each variable
    for isolation in isolations:
        for i,variable in enumerate(variables):
            xs  = data[:, [ninputs+1+i]].astype(np.float32).ravel()
            graphs.append(efficiency.efficiency_graph(pass_function=(lambda x:np.less(x[1],isolation.predict(x[0]))), function_inputs=(inputs,targets), xs=xs))
            graphs[-1].SetName(isolation.name+'_'+variable+'_test')
    return graphs


def optimize_background_rejection(effs, isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputnames=['abs(ieta)','ntt'], targetname='iso'):
    # Compute signal efficiencies
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    #branches.extend(variables)
    data = root2array(signalfile, treename=signaltree, branches=branches, selection='et>10')
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    signal_efficiencies = [efficiency.efficiency_inclusive(pass_function=(lambda x:np.less(x[1],iso.predict(x[0]))), function_inputs=(inputs,targets))[0] for iso in isolations]
    # Compute background efficiencies
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    #branches.extend(variables)
    data = root2array(backgroundfile, treename=backgroundtree, branches=branches, selection='et>10')
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    background_efficiencies = [efficiency.efficiency_inclusive(pass_function=(lambda x:np.less(x[1],iso.predict(x[0]))), function_inputs=(inputs,targets))[0] for iso in isolations]
    # Compute efficiency gradients
    effs_diff = np.ediff1d(effs)
    signal_efficiencies_diff = np.ediff1d(signal_efficiencies)
    background_efficiencies_diff = np.ediff1d(background_efficiencies)
    signal_efficiencies_diff = np.divide(signal_efficiencies_diff, effs_diff)
    background_efficiencies_diff = np.divide(background_efficiencies_diff, effs_diff)
    # Interpolate and find points where background efficiency gradient > signal efficiency gradient (with some tolerance)
    interp_x = np.linspace(np.amin(effs[1:]), np.amax(effs[1:]), 1000)
    interp_signal = np.interp(interp_x, effs[1:], signal_efficiencies_diff)
    interp_background = np.interp(interp_x, effs[1:], background_efficiencies_diff)
    optimal_points = np.argwhere(np.greater(interp_background-0.05, interp_signal)).ravel() # Use a tolerance of 0.02 in case of fluctuations
    if len(optimal_points)==0:
        print 'WARNING: no working point found where the efficiency gradient is larger for background than for signal'
    # Find optimal point with smallest efficiency
    ## Compute spacing between points, and select those with an efficiency separation > 2%
    optimal_discontinuities = np.argwhere(np.ediff1d(interp_x[optimal_points])>0.02).ravel()
    ## Select the point with the largest efficiency
    optimal_index = np.amax(optimal_discontinuities)+1 if len(optimal_discontinuities)>0 else 0
    optimal_point = interp_x[optimal_points[optimal_index]]
    # Create graphs
    signal_efficiencies_diff_graph = Graph(len(effs)-1)
    background_efficiencies_diff_graph = Graph(len(effs)-1)
    optimal_points_graph = Graph(len(optimal_points))
    fill_graph(signal_efficiencies_diff_graph, np.column_stack((effs[1:], signal_efficiencies_diff)))
    fill_graph(background_efficiencies_diff_graph, np.column_stack((effs[1:], background_efficiencies_diff)))
    fill_graph(optimal_points_graph, np.column_stack((interp_x[optimal_points], interp_signal[optimal_points])))
    signal_efficiencies_diff_graph.SetName('efficiencies_signal')
    background_efficiencies_diff_graph.SetName('efficiencies_background')
    optimal_points_graph.SetName('signal_background_optimal_points')
    return signal_efficiencies_diff_graph, background_efficiencies_diff_graph, optimal_points_graph, optimal_point


def main(signalfile, signaltree, backgroundfile, backgroundtree, outputdir, name, test=False, inputs=['abs(ieta)','ntt'], target='iso', pileupref='rho'):
    # Compute isolation cuts for efficiencies from 0.2 to 1 with smaller steps for larger efficiencies
    effs = np.arange(0.2,0.5,0.05)
    effs = np.concatenate((effs,np.arange(0.5,0.85,0.02)))
    effs = np.concatenate((effs,np.arange(0.85,0.999,0.01)))
    #effs = np.arange(0.6,1.,0.1) # for tests
    version = batch_launcher.job_version(outputdir)
    workingdir = outputdir+'/'+version
    # Train isolation cuts
    eg_isolations = train_isolation_workingpoints(effs, signalfile, signaltree, outputdir, version, name, test, inputs, target, pileupref)
    with root_open(workingdir+'/'+name+'.root', 'recreate') as output_file:
        # Save isolation cuts in TH2s
        for eff,eg_isolation_cuts in zip(effs,eg_isolations):
            histo = function2th2(eg_isolation_cuts.predict, quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]])
            histo.SetName(name+'_'+str(eff))
            histo.Write()
        # Test isolation cuts vs offline variables
        print '> Checking efficiencies vs offline variables'
        graphs = test_isolation_workingpoints(effs, eg_isolations, signalfile, signaltree, inputs, target)
        for graph in graphs:
            graph.Write()
        # Optimize signal efficiency vs background rejection
        print '> Optimizing signal efficiency vs background rejection'
        signal_efficiencies_diff_graph, background_efficiencies_diff_graph, optimal_points_graph, optimal_point = optimize_background_rejection(effs, eg_isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputs, target)
        print '   Choosing working point', optimal_point
        signal_efficiencies_diff_graph.Write()
        background_efficiencies_diff_graph.Write()
        optimal_points_graph.Write()



if __name__=='__main__':
    import optparse
    usage = 'usage: python %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--signalfile', dest='signal_file', help='Input file', default='tree.root')
    parser.add_option('--signaltree', dest='signal_tree', help='Tree in the input file', default='tree')
    parser.add_option('--backgroundfile', dest='background_file', help='Input file', default='tree.root')
    parser.add_option('--backgroundtree', dest='background_tree', help='Tree in the input file', default='tree')
    parser.add_option('--outputdir', dest='output_dir', help='Output directory', default='./')
    parser.add_option('--name', dest='name', help='Name used for the results', default='egamma_isolation')
    parser.add_option('--test', action="store_true", dest='test', help='Flag to test regression on a test sample', default=False)
    parser.add_option('--inputs', dest='inputs', help='List of input variables of the form "var1,var2,..."', default='abs(ieta),ntt')
    parser.add_option('--pileupref', dest='pileup_ref', help='Reference variable used for pile-up', default='rho')
    parser.add_option('--target', dest='target', help='Target variable', default='iso')
    (opt, args) = parser.parse_args()
    inputs = opt.inputs.replace(' ','').split(',')
    main(signalfile=opt.signal_file, signaltree=opt.signal_tree,
        backgroundfile=opt.background_file, backgroundtree=opt.background_tree,
         outputdir=opt.output_dir, name=opt.name, test=opt.test, inputs=inputs, target=opt.target, pileupref=opt.pileup_ref)

