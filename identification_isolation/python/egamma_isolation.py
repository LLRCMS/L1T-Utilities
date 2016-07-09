from object_conversions.conversion_to_histo import function2th2
from batch import batch_launcher
from identification_isolation import quantile_regression, correlations, efficiency

import copy
import os

import numpy as np
from sklearn.externals import joblib

from rootpy.plotting import Hist2D
from rootpy.io import root_open
from root_numpy import root2array


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


def test_efficiencies(isolations, inputfile, tree, inputnames, targetname, variables):
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
            graphs[-1].SetName(isolation.name+'_'+variable)
    return graphs

def main(inputfile, tree, outputdir, name, test=False, inputs=['abs(ieta)','ntt'], target='iso', pileupref='rho'):
    # Compute isolation cuts for efficiencies from 0.2 to 1 with smaller steps for larger efficiencies
    effs = np.arange(0.2,0.5,0.05)
    effs = np.concatenate((effs,np.arange(0.5,0.85,0.02)))
    effs = np.concatenate((effs,np.arange(0.85,0.999,0.01)))
    #effs = np.arange(0.6,1.,0.1)
    print '> Deriving {0}->{1} map'.format(pileupref, inputs[1])
    pu_regression = correlations.fit_linear(inputfile, tree, pileupref, inputs[1], test=False)
    print '> Deriving isolation working points'
    version = batch_launcher.job_version(outputdir)
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
    with root_open(workingdir+'/'+name+'.root', 'recreate') as output_file:
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
            # Create TH2 filled with isolation cuts
            histo = function2th2(eg_isolation_cuts.predict, quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]])
            histo.SetName(name+'_'+str(eff))
            histo.Write()
        # Check efficiencies
        print '> Checking efficiencies vs offline variables'
        graphs = test_efficiencies(isolations=eg_isolations, inputfile=inputfile, tree=tree, inputnames=inputs, targetname=target, variables=['offl_eta','offl_pt', 'rho', 'npv'])
        for graph in graphs:
            graph.Write()



if __name__=='__main__':
    import optparse
    usage = 'usage: python %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--inputfile', dest='input_file', help='Input file', default='tree.root')
    parser.add_option('--tree', dest='tree_name', help='Tree in the input file', default='tree')
    parser.add_option('--outputdir', dest='output_dir', help='Output directory', default='./')
    parser.add_option('--name', dest='name', help='Name used for the results', default='egamma_isolation')
    parser.add_option('--test', action="store_true", dest='test', help='Flag to test regression on a test sample', default=False)
    parser.add_option('--inputs', dest='inputs', help='List of input variables of the form "var1,var2,..."', default='abs(ieta),ntt')
    parser.add_option('--pileupref', dest='pileup_ref', help='Reference variable used for pile-up', default='rho')
    parser.add_option('--target', dest='target', help='Target variable', default='iso')
    (opt, args) = parser.parse_args()
    inputs = opt.inputs.replace(' ','').split(',')
    main(inputfile=opt.input_file, tree=opt.tree_name, outputdir=opt.output_dir, name=opt.name, test=opt.test, inputs=inputs, target=opt.target, pileupref=opt.pileup_ref)
