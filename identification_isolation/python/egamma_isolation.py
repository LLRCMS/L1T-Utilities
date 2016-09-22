from batch import batch_launcher
from identification_isolation import quantile_regression, correlations, efficiency
from identification_isolation.tests import test_efficiency
from utilities.root_utilities import function2th2, function2th3, events2th3, graph2array
from identification_isolation.cut_functions import RegressionWithInputMapping, CombinedWorkingPoints 
import rate

import sys
import copy
import os
import pickle

import numpy as np
from sklearn.externals import joblib


from rootpy.plotting import Hist, Hist2D, Graph
from rootpy.io import root_open
from root_numpy import root2array, hist2array, array2hist, fill_graph, fill_hist, evaluate




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


def train_isolation_workingpoints(steps, effs, inputfile, tree, outputdir, version, name, inputs=['abs(ieta)','ntt'], target='iso', pileupref='rho'):
    workingdir = outputdir+'/'+version
    # Replacing L1 pile-up variable (ntt) with the reference pile-up variable (rho)
    regression_inputs = copy.deepcopy(inputs)
    regression_inputs[1] = pileupref
    parameters = iso_parameters(inputfile, tree, name, regression_inputs, target, effs, test=False)
    if steps.train_workingpoints:
        print '> Deriving isolation working points'
        batch_launcher.main(workingdir=outputdir,
                            exe='python {}/identification_isolation/python/quantile_regression.py'.format(os.environ['L1TSTUDIES_BASE']),
                            pars=parameters)
    if steps.fit_ntt_vs_rho:
        print '> Deriving {0}->{1} map'.format(pileupref, inputs[1])
        pu_regression = correlations.fit_linear(inputfile, tree, pileupref, inputs[1], test=False)
        fit_parameters = [pu_regression.intercept_, pu_regression.coef_[0]]
        pickle.dump(fit_parameters, open(workingdir+'/ntt_vs_rho_regression.pkl', 'wb'))
    else:
        fit_parameters = pickle.load(open(workingdir+'/ntt_vs_rho_regression.pkl', 'rb'))
    if steps.train_workingpoints:
        print '> Waiting working point trainings...'
        batch_launcher.wait_jobs(workingdir, wait=5)
        print '  ... trainings done'
    print '> Applying {0}->{1} map'.format(pileupref, inputs[1])
    eg_isolations = []
    for i,pars in enumerate(parameters):
        # Load saved isolation regression
        eff = float(pars['eff'])
        result_dir = workingdir+'/'+name+'_{}'.format(eff*100)
        iso_regression = joblib.load(result_dir+'/'+name+'.pkl')
        # Apply rho->ntt linear mapping on isolation regression
        a = fit_parameters[0]
        b = fit_parameters[1]
        eg_isolation_cuts = RegressionWithInputMapping(name='eg_iso_'+pars['eff'], iso_regression=iso_regression, input_mappings={1:(lambda x:max(0.,(x-a)/b))})
        eg_isolations.append(eg_isolation_cuts)
    return eg_isolations



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
    signal_efficiencies_diff_3 = np.convolve(signal_efficiencies_diff, np.repeat(1.0, 3.)/3., 'valid')
    signal_efficiencies_diff_2 = np.convolve(signal_efficiencies_diff, np.repeat(1.0, 2.)/2., 'valid')
    signal_efficiencies_diff = np.append(signal_efficiencies_diff_3, [signal_efficiencies_diff_2[-1],signal_efficiencies_diff[-1]])
    background_efficiencies_diff_3 = np.convolve(background_efficiencies_diff, np.repeat(1.0, 3.)/3., 'valid')
    background_efficiencies_diff_2 = np.convolve(background_efficiencies_diff, np.repeat(1.0, 2.)/2., 'valid')
    background_efficiencies_diff = np.append(background_efficiencies_diff_3, [background_efficiencies_diff_2[-1],background_efficiencies_diff[-1]])
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
    data = root2array(signalfile, treename=signaltree, branches=branches, selection=cut)
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    signal_efficiencies = np.array([efficiency.efficiency_inclusive(pass_function=(lambda x:np.less(x[1],iso.predict(x[0]))), function_inputs=(inputs,targets))[0] for iso in isolations])
    # Compute background efficiencies
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    data = root2array(backgroundfile, treename=backgroundtree, branches=branches, selection=cut)
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    background_efficiencies = np.array([efficiency.efficiency_inclusive(pass_function=(lambda x:np.less(x[1],iso.predict(x[0]))), function_inputs=(inputs,targets))[0] for iso in isolations])
    proba_signal = np.ones(signal_efficiencies.shape)
    proba_background = np.ones(background_efficiencies.shape)
    return find_best_working_point(effs, signal_efficiencies, background_efficiencies, proba_signal, proba_background)


# TODO: merge the two optimize functions into a single one
def optimize_background_rejection_vs_ieta(effs, isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputnames=['abs(ieta)','ntt'], targetname='iso', cut='et>10'):
    ieta_binning = [0.5, 3.5, 6.5, 9.5, 13.5, 18.5, 24.5, 27.5]
    # Compute signal efficiencies
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    data = root2array(signalfile, treename=signaltree, branches=branches, selection=cut)
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
    data = root2array(backgroundfile, treename=backgroundtree, branches=branches, selection=cut)
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


def single_slope_relaxation_vs_pt(optimal_points_vs_ieta, threshold, eff_min=0.4,max_et=110):
    points_vs_ieta_pt = Hist2D(np.array(optimal_points_vs_ieta.GetXaxis().GetXbins()), 200, 0.5, 200.5)
    for bx in points_vs_ieta_pt.bins_range(0):
        eff_ref = optimal_points_vs_ieta[bx].value
        for by in points_vs_ieta_pt.bins_range(1):
            et = points_vs_ieta_pt.GetYaxis().GetBinCenter(by)
            eff = (1.-eff_ref)/(max_et-threshold)*(et-threshold) + eff_ref
            eff = max(eff_min, min(1, eff))
            points_vs_ieta_pt[bx,by].value = eff
    return points_vs_ieta_pt

def double_slope_relaxation_vs_pt(optimal_points_vs_ieta_low, optimal_points_vs_ieta_high, threshold_low, threshold_high, eff_min=0.4,max_et=110):
    points_vs_ieta_pt = Hist2D(np.array(optimal_points_vs_ieta_low.GetXaxis().GetXbins()), 200, 0.5, 200.5)
    for bx in points_vs_ieta_pt.bins_range(0):
        eff_ref_low = optimal_points_vs_ieta_low[bx].value
        eff_ref_high = optimal_points_vs_ieta_high[bx].value
        for by in points_vs_ieta_pt.bins_range(1):
            et = points_vs_ieta_pt.GetYaxis().GetBinCenter(by)
            eff_high = (1.-eff_ref_high)/(max_et-threshold_high)*(et-threshold_high) + eff_ref_high
            eff_low = (eff_ref_high-eff_ref_low)/(threshold_high-threshold_low)*(et-threshold_low) + eff_ref_low
            eff = eff_high
            if et<threshold_high:
                eff = eff_low
            eff = max(eff_min, min(1, eff))
            points_vs_ieta_pt[bx,by].value = eff
    return points_vs_ieta_pt


def main(parameters):
    # Compute isolation cuts for efficiencies from 0.2 to 1 with smaller steps for larger efficiencies
    effs = np.arange(0.2,0.5,0.05)
    effs = np.concatenate((effs,np.arange(0.5,0.85,0.02)))
    effs = np.concatenate((effs,np.arange(0.85,0.999,0.01)))
    # if no version specified, automatically set version number
    if parameters.version is 'automatic':
        # if training of the working points requested
        # create a new version
        if parameters.steps.train_workingpoints:
            version = batch_launcher.job_version(parameters.working_directory)
        # else, use the last version available
        else:
            version = batch_launcher.latest_version(parameters.working_directory)
            if version is '':
                raise StandardError('Cannot find already trained working points')
    else:
        version = parameters.version
    workingdir = parameters.working_directory+'/'+version
    inputs = [
        parameters.variables.ieta,
        parameters.variables.ntt,
    ]
    target = parameters.variables.iso
    pileupref = parameters.variables.rho
    # Train isolation cuts
    eg_isolations = train_isolation_workingpoints(parameters.steps,
                                                  effs,
                                                  parameters.signal_file,
                                                  parameters.signal_tree,
                                                  parameters.working_directory,
                                                  version,
                                                  parameters.name,
                                                  inputs,
                                                  target,
                                                  pileupref)
    with root_open(workingdir+'/'+parameters.name+'.root', 'recreate') as output_file:
        # Save isolation cuts in TH2s
        for eff,eg_isolation_cuts in zip(effs,eg_isolations):
            histo = function2th2(eg_isolation_cuts.predict, quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]])
            histo.SetName(parameters.name+'_'+str(eff))
            histo.Write()
        # Test isolation cuts vs offline variables
        if parameters.steps.test_workingpoints:
            print '> Checking efficiencies vs offline variables'
            graphs = test_efficiency(functions=[(lambda x,isolation=iso:np.less(x[:,[len(inputs)]].ravel(),isolation.predict(x[:,range(len(inputs))]))) for iso in eg_isolations], \
                                     function_inputs=inputs+[target],\
                                     variables=['offl_eta','offl_pt', 'rho', 'npv'],\
                                     inputfile=parameters.signal_file,\
                                     tree=parameters.signal_tree,\
                                     selection='et>0'\
                                    )
            for graph in graphs:
                graph.Write()
        print '> Applying eta/et efficiency shape'
        eg_isolation_eta_et = CombinedWorkingPoints(np.append(effs,[1.]),
                                                    [iso.predict for iso in eg_isolations]+[lambda x:np.full(x.shape[0],9999.)],
                                                    parameters.eta_pt_optimization.eta_pt_efficiency_shapes)
        print '> Compress input variables'
        branches = [
            parameters.variables.ieta,    
            parameters.variables.ntt,
            parameters.variables.et,
        ]
        data = root2array(parameters.signal_file,
                          treename=parameters.signal_tree,
                          branches=branches,
                          selection='et>0')
        data = data.view((np.float64, len(data.dtype.names))).astype(np.float32)
        iso_cuts = eg_isolation_eta_et.value(data[:,[0,1]],data[:,[0,2]])
        eg_isolation_compressed = events2th3(data, iso_cuts,
                                             (parameters.compression.eta,),
                                             (parameters.compression.ntt,),
                                             (parameters.compression.et,))
        eg_isolation_compressed.SetName('isolation_compressed_')
        eg_isolation_compressed.Write()
        graphs_compressed = test_efficiency(functions=(lambda x: np.less(x[:,[3]].ravel(),evaluate(eg_isolation_compressed, x[:,range(3)]))), \
                                      function_inputs=branches+[parameters.variables.iso],\
                                      variables=['offl_eta','offl_pt', 'rho', 'npv'],\
                                      inputfile=parameters.signal_file,\
                                      tree=parameters.signal_tree,\
                                      selection='et>0'\
                                     )
        for graph in graphs_compressed:
            graph.Write()


if __name__=='__main__':
    import optparse
    import importlib
    usage = 'usage: python %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--cfg', dest='parameter_file', help='Python file containing the definition of parameters ', default='pars.py')
    (opt, args) = parser.parse_args()
    current_dir = os.getcwd();
    sys.path.append(current_dir)
    # Remove the extension of the python file before module loading
    if opt.parameter_file[-3:]=='.py': opt.parameter_file = opt.parameter_file[:-3]
    parameters = importlib.import_module(opt.parameter_file).parameters
    main(parameters)

