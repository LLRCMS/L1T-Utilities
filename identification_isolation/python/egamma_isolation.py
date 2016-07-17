from batch import batch_launcher
from identification_isolation import quantile_regression, correlations, efficiency
from object_conversions.conversion_to_histo import function2th2, function2th3

import copy
import os

import numpy as np
from sklearn.externals import joblib


from rootpy.plotting import Hist, Hist2D, Graph
from rootpy.io import root_open
from root_numpy import root2array, hist2array, array2hist, fill_graph, fill_hist, evaluate




def graph2array(graph):
    xs = np.array([graph.GetX()[p] for p in range(graph.GetN())])
    ys = np.array([graph.GetY()[p] for p in range(graph.GetN())])
    return np.column_stack((xs,ys))

def find_closest(a, target):
    # a must be sorted
    idx = a.searchsorted(target)
    idx = np.clip(idx, 1, len(a)-1)
    left = a[idx-1]
    right = a[idx]
    idx -= target - left < right - target
    return idx


# Compound of multivariate isolation cuts and input mappings
class IsolationCuts:
    def __init__(self, iso_regression, input_mappings, name='isolation'):
        self.name = name
        self.iso_regression = iso_regression
        # dictionary input index -> function to be applied on inputs
        self.input_mappings = input_mappings
        # Vectorize the functions such that they can take arrays as input
        for index, mapping in self.input_mappings.items():
            self.input_mappings[index] = np.vectorize(mapping)

    def predict(self, values):
        #print 'In IsolationCuts.predict()'
        # Apply input mappings
        mapped_inputs = np.array(values, dtype=np.float64)
        for index,mapping in self.input_mappings.items():
            # Apply mapping on column 'index'
            mapped_inputs_i = mapping(mapped_inputs[:,[index]])
            # Replace column 'index' with mapped inputs
            mapped_inputs = np.delete(mapped_inputs, index, axis=1)
            mapped_inputs = np.insert(mapped_inputs, [index], mapped_inputs_i, axis=1)
        # Apply iso regression on mapped inputs
        output = self.iso_regression.predict(mapped_inputs)
        #print 'Out IsolationCuts.predict()'
        return output


class IsolationCombinedCuts:
    def __init__(self, working_points, functions, efficiency_map):
        print working_points
        print functions
        efficiency_array = hist2array(efficiency_map)
        print 'Creating combined iso cuts with efficiencies', efficiency_array
        working_points_indices = find_closest(working_points, efficiency_array)
        print working_points_indices
        self.function_index_map = efficiency_map.empty_clone()
        array2hist(working_points_indices, self.function_index_map)
        self.indices = working_points_indices
        self.functions = functions
        self.dim = len(efficiency_array.shape)
        print self.dim

    def value(self, inputs, map_positions):
        print 'In IsolationCombinedCuts.value()'
        print '  Find indices'
        upper_bounds = [self.function_index_map.bounds(axis)[1]-1e-3 for axis in range(len(self.function_index_map.axes))]
        map_positions_no_overflow = np.apply_along_axis(lambda x:np.minimum(x,upper_bounds), 1, map_positions)
        if self.dim==1: map_positions_no_overflow = map_positions_no_overflow.ravel()
        indices = evaluate(self.function_index_map, map_positions_no_overflow).astype(np.int32)
        print map_positions
        print indices
        print '  Compute values'
        outputs = []
        for i,function in enumerate(self.functions):
            if i in self.indices:
                print '    Index',i
                outputs.append(function(inputs))
            else:
                outputs.append(np.array([]))
        #output = [self.functions[index]([input]) for index,input in zip(indices,inputs)]
        print '  Choose indices'
        output = np.zeros(len(indices))
        for i,index in enumerate(indices):
            #print i, len(indices), len(inputs), index, len(outputs[index])
            output[i] = outputs[index][i]
        print 'Out IsolationCombinedCuts.value()'
        return output







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
    #batch_launcher.main(workingdir=outputdir,
                        #exe='python {}/identification_isolation/python/quantile_regression.py'.format(os.environ['L1TSTUDIES_BASE']),
                        #pars=parameters)
    #print '> Waiting batch jobs to finish...'
    #batch_launcher.wait_jobs(workingdir, wait=5)
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
        eg_isolation_cuts = IsolationCuts(name='eg_iso_'+pars['eff'], iso_regression=iso_regression, input_mappings={1:(lambda x:max(0.,(x-a)/b))})
        eg_isolations.append(eg_isolation_cuts)
    return eg_isolations



def test_combined_isolation(isolation, inputfile, tree, inputnames=['abs(ieta)','ntt'], targetname='iso', variables=['offl_eta','offl_pt', 'rho', 'npv']):
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
    # Compute efficiencies for each variable
    for i,variable in enumerate(variables):
        xs  = data[:, [ninputs+1+i]].astype(np.float32).ravel()
        graphs.append(efficiency.efficiency_graph(pass_function=(lambda x: np.less(x[1],isolation.value(x[0],x[0][:,[0]]))), function_inputs=(inputs,targets), xs=xs))
        #graphs.append(efficiency.efficiency_graph(pass_function=(lambda x: np.less(x[1],evaluate(isolation,x[0]))), function_inputs=(inputs,targets), xs=xs))
        graphs[-1].SetName('combined_'+variable+'_test')
    return graphs

def test_combined_isolation_pt(isolation, inputfile, tree, inputnames=['abs(ieta)','ntt','et'], targetname='iso', variables=['offl_eta','offl_pt', 'rho', 'npv']):
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
    # Compute efficiencies for each variable
    for i,variable in enumerate(variables):
        xs  = data[:, [ninputs+1+i]].astype(np.float32).ravel()
        graphs.append(efficiency.efficiency_graph(pass_function=(lambda x: np.less(x[1],isolation.value(x[0][:,[0,1]],x[0][:,[0,2]]))), function_inputs=(inputs,targets), xs=xs))
        #graphs.append(efficiency.efficiency_graph(pass_function=(lambda x: np.less(x[1],evaluate(isolation,x[0]))), function_inputs=(inputs,targets), xs=xs))
        graphs[-1].SetName('combined_pt_'+variable+'_test')
    return graphs

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

def find_best_working_point(effs, signal_efficiencies, background_efficiencies, signal_probabilities, background_probabilities):
    # Compute ratios of bacground and signal probabilities and truncate the array
    # to match the gradient array size
    probability_ratios = background_probabilities/signal_probabilities
    probability_ratios = probability_ratios[1:]
    # Compute efficiency gradients
    effs_diff = np.ediff1d(effs)
    signal_efficiencies_diff = np.ediff1d(signal_efficiencies)
    background_efficiencies_diff = np.ediff1d(background_efficiencies)
    signal_efficiencies_diff = np.divide(signal_efficiencies_diff, effs_diff)
    background_efficiencies_diff = np.divide(background_efficiencies_diff, effs_diff)
    # Apply the signal and background probabilities in order to weight the efficiency gradients
    # If it is more likely to have background than signal in this bin, then the background efficiency gradient
    # will be increased accordingly
    background_efficiencies_diff = background_efficiencies_diff * probability_ratios
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



def optimize_background_rejection(effs, isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputnames=['abs(ieta)','ntt'], targetname='iso', cut='et>10'):
    # Compute signal efficiencies
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    #branches.extend(variables)
    data = root2array(signalfile, treename=signaltree, branches=branches, selection=cut)
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    signal_efficiencies = np.array([efficiency.efficiency_inclusive(pass_function=(lambda x:np.less(x[1],iso.predict(x[0]))), function_inputs=(inputs,targets))[0] for iso in isolations])
    # Compute background efficiencies
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    #branches.extend(variables)
    data = root2array(backgroundfile, treename=backgroundtree, branches=branches, selection=cut)
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    background_efficiencies = np.array([efficiency.efficiency_inclusive(pass_function=(lambda x:np.less(x[1],iso.predict(x[0]))), function_inputs=(inputs,targets))[0] for iso in isolations])
    proba_signal = np.ones(signal_efficiencies.shape)
    proba_background = np.ones(background_efficiencies.shape)
    return find_best_working_point(effs, signal_efficiencies, background_efficiencies, proba_signal, proba_background)


def optimize_background_rejection_vs_ieta(effs, isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputnames=['abs(ieta)','ntt'], targetname='iso'):
    #ieta_binning = np.arange(0.5,28.5,1)
    ieta_binning = [0.5, 3.5, 6.5, 9.5, 13.5, 18.5, 22.5, 27.5]
    # Compute signal efficiencies
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    data = root2array(signalfile, treename=signaltree, branches=branches, selection='et>10')
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    xs  = data[:, [0]].astype(np.float32).ravel()
    # fill signal ieta histogram and normalize to 1
    histo_signal = Hist(ieta_binning)
    fill_hist(histo_signal, xs)
    #histo_signal.Scale(1./histo_signal.integral(overflow=True))
    # signal_efficiencies is a 2D array 
    # The first dimension corresponds to different ieta values
    # The second dimension corresponds to different working points
    signal_efficiencies = [graph2array(efficiency.efficiency_graph(pass_function=(lambda x:np.less(x[1],iso.predict(x[0]))), function_inputs=(inputs,targets), xs=xs, bins=ieta_binning))[:,[1]].ravel() for iso in isolations]
    signal_efficiencies = np.column_stack(signal_efficiencies)
    # Compute background efficiencies
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    data = root2array(backgroundfile, treename=backgroundtree, branches=branches, selection='et>10')
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    xs  = data[:, [0]].astype(np.float32).ravel()
    # fill background ieta histogram and normalize to 1
    histo_background = Hist(ieta_binning)
    fill_hist(histo_background, xs)
    #histo_background.Scale(1./histo_background.integral(overflow=True))
    # background_efficiencies is a 2D array 
    # The first dimension corresponds to different ieta values
    # The second dimension corresponds to different working points
    background_efficiencies = [graph2array(efficiency.efficiency_graph(pass_function=(lambda x:np.less(x[1],iso.predict(x[0]))), function_inputs=(inputs,targets), xs=xs, bins=ieta_binning))[:,[1]].ravel() for iso in isolations]
    background_efficiencies = np.column_stack(background_efficiencies)
    signal_efficiencies_diff_graphs = []
    background_efficiencies_diff_graphs = []
    optimal_points_graphs = []
    optimal_points = []
    # compute best working point for each ieta (loop on ieta)
    for i,(signal_effs,background_effs) in enumerate(zip(signal_efficiencies, background_efficiencies)):
        # Compute the probability of signal in this ieta bin for the different efficiency points
        # It is assumed that the cut is applied only in this bin, all the other bins keep the same number of entries
        n_i = histo_signal[i+1].value 
        n_tot = histo_signal.integral(overflow=True)
        proba_signal = np.array([n_i*eff/(n_tot-n_i*(1.-eff)) for eff in signal_effs])
        # Same as above, but for background
        n_i = histo_background[i+1].value 
        n_tot = histo_background.integral(overflow=True)
        proba_background = np.array([n_i*eff/(n_tot-n_i*(1.-eff)) for eff in background_effs])
        signal_efficiencies_diff_graph, background_efficiencies_diff_graph, optimal_points_graph, optimal_point = find_best_working_point(effs, signal_effs, background_effs, proba_signal, proba_background)
        signal_efficiencies_diff_graph.SetName('efficiencies_signal_ieta_{}'.format(i))
        background_efficiencies_diff_graph.SetName('efficiencies_background_ieta_{}'.format(i))
        optimal_points_graph.SetName('signal_background_optimal_points_ieta_{}'.format(i))
        signal_efficiencies_diff_graphs.append(signal_efficiencies_diff_graph)
        background_efficiencies_diff_graphs.append(background_efficiencies_diff_graph)
        optimal_points_graphs.append(optimal_points_graph)
        optimal_points.append(optimal_point)
    optimal_points_histo = Hist(ieta_binning)
    array2hist(optimal_points, optimal_points_histo)
    return signal_efficiencies_diff_graphs, background_efficiencies_diff_graphs, optimal_points_graphs, optimal_points_histo

def relax_efficiency_vs_pt(optimal_points_vs_ieta, threshold, eff_min=0.4,max_et=110):
    points_vs_ieta_pt = Hist2D(np.array(optimal_points_vs_ieta.GetXaxis().GetXbins()), 200, 0.5, 200.5)
    for bx in points_vs_ieta_pt.bins_range(0):
        eff_ref = optimal_points_vs_ieta[bx].value
        for by in points_vs_ieta_pt.bins_range(1):
            et = points_vs_ieta_pt.GetYaxis().GetBinCenter(by)
            eff = (1.-eff_ref)/(max_et-threshold)*(et-threshold) + eff_ref
            eff = max(eff_min, min(1, eff))
            points_vs_ieta_pt[bx,by].value = eff
    return points_vs_ieta_pt


def main(signalfile, signaltree, backgroundfile, backgroundtree, outputdir, name, test=False, inputs=['abs(ieta)','ntt'], target='iso', pileupref='rho'):
    # Compute isolation cuts for efficiencies from 0.2 to 1 with smaller steps for larger efficiencies
    effs = np.arange(0.2,0.5,0.05)
    effs = np.concatenate((effs,np.arange(0.5,0.85,0.02)))
    effs = np.concatenate((effs,np.arange(0.85,0.999,0.01)))
    #effs = np.arange(0.6,1.,0.1) # for tests
    version = batch_launcher.job_version(outputdir)
    version = 'v_5_2016-07-13'
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
        #print '> Checking efficiencies vs offline variables'
        #graphs = test_isolation_workingpoints(effs, eg_isolations, signalfile, signaltree, inputs, target)
        #for graph in graphs:
            #graph.Write()
        # Optimize signal efficiency vs background rejection
        print '> Optimizing signal efficiency vs background rejection'
        #for et_cut in [10, 15, 20, 30, 40, 50]:
        et_cut = 30
        signal_efficiencies_diff_graph, background_efficiencies_diff_graph, optimal_points_graph, optimal_point = optimize_background_rejection(effs, eg_isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputs, target, cut='et>{}'.format(et_cut))
        print '   Best inclusive working point', optimal_point
        signal_efficiencies_diff_graph.SetName(signal_efficiencies_diff_graph.GetName()+'_et_{}'.format(et_cut))
        background_efficiencies_diff_graph.SetName(background_efficiencies_diff_graph.GetName()+'_et_{}'.format(et_cut))
        optimal_points_graph.SetName(optimal_points_graph.GetName()+'_et_{}'.format(et_cut))
        signal_efficiencies_diff_graph.Write() 
        background_efficiencies_diff_graph.Write()
        optimal_points_graph.Write()
        signal_efficiencies_diff_graphs, background_efficiencies_diff_graphs, optimal_points_graphs, optimal_points = optimize_background_rejection_vs_ieta(effs, eg_isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputs, target)
        combined_cuts = IsolationCombinedCuts(effs, [iso.predict for iso in eg_isolations], optimal_points)
        #combined_cuts_histo = function2th2(lambda x: combined_cuts.value(x,x[:,[0]].ravel()), quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]])
        combined_cuts_histo = function2th2(lambda x: combined_cuts.value(x,x[:,[0]]), quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]])
        print '> Applying pt relaxation'
        points_vs_pt_ieta = relax_efficiency_vs_pt(optimal_points, 60)
        combined_cuts_pt = IsolationCombinedCuts(np.append(effs,[1.]), [iso.predict for iso in eg_isolations]+[lambda x:np.full(x.shape[0],9999.)], points_vs_pt_ieta)
        #print '> Converting pt relaxed into histo'
        #combined_cuts_pt_histo = function2th3(lambda x: combined_cuts_pt.value(x[:,[0,1]],x[:,[0,2]]), quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]], quantile_regression.binning['et'])
        print '> Testing combined iso cuts'
        #graphs_comb = test_combined_isolation(combined_cuts_histo, signalfile, signaltree, inputs, target)
        graphs_comb = test_combined_isolation(combined_cuts, signalfile, signaltree, inputs, target)
        for graph in graphs_comb:
            graph.Write()
        print '> Testing combined iso cuts vs pt'
        graphs_comb_pt = test_combined_isolation_pt(combined_cuts_pt, signalfile, signaltree, inputs+['et'], target)
        for graph in graphs_comb_pt:
            graph.Write()
        combined_cuts_histo.SetName('optimal_cuts')
        combined_cuts_histo.Write()
        print '   Best working points vs |ieta|', hist2array(optimal_points)
        for graph in signal_efficiencies_diff_graphs: 
            graph.Write()
        for graph in background_efficiencies_diff_graphs:
            graph.Write()
        for graph in optimal_points_graphs:
            graph.Write()
        optimal_points.SetName('optimal_points_vs_ieta')
        optimal_points.Write()
        points_vs_pt_ieta.SetName('points_vs_pt_ieta')
        points_vs_pt_ieta.Write()



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

