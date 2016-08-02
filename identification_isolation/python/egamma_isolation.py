from batch import batch_launcher
from identification_isolation import quantile_regression, correlations, efficiency
from object_conversions.conversion_to_histo import function2th2, function2th3, events2th3
from utilities.root_utilities import graph2array
from identification_isolation.cut_functions import RegressionWithInputMapping, CombinedWorkingPoints 
import rate

import copy
import os

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
        eg_isolation_cuts = RegressionWithInputMapping(name='eg_iso_'+pars['eff'], iso_regression=iso_regression, input_mappings={1:(lambda x:max(0.,(x-a)/b))})
        eg_isolations.append(eg_isolation_cuts)
    return eg_isolations








def test_combined_isolation(isolation, inputfile, tree, inputnames=['abs(ieta)','ntt'], targetname='iso', variables=['offl_eta','offl_pt', 'rho', 'npv']):
    # Retrieve data from tree
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    branches.extend(variables)
    data = root2array(inputfile, treename=tree, branches=branches, selection='et>0')
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

def test_combined_isolation_pt(isolation, inputfile, tree, inputnames=['abs(ieta)','ntt','et_raw'], targetname='iso', variables=['offl_eta','offl_pt', 'rho', 'npv']):
    # Retrieve data from tree
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    branches.extend(variables)
    data = root2array(inputfile, treename=tree, branches=branches, selection='et>0')
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

def test_combined_isolation_pt_compressed(isolation, inputfile, tree, inputnames=['abs(ieta)','ntt','et_raw'], targetname='iso', variables=['offl_eta','offl_pt', 'rho', 'npv']):
    # Retrieve data from tree
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    branches.extend(variables)
    data = root2array(inputfile, treename=tree, branches=branches, selection='et>0')
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    graphs = []
    # Compute efficiencies for each variable
    for i,variable in enumerate(variables):
        xs  = data[:, [ninputs+1+i]].astype(np.float32).ravel()
        graphs.append(efficiency.efficiency_graph(pass_function=(lambda x: np.less(x[1],evaluate(isolation, x[0]))), function_inputs=(inputs,targets), xs=xs))
        graphs[-1].SetName('combined_pt_compressed_'+variable+'_test')
    return graphs

def test_current_isolation(inputfile, tree, iso = 'iso_pass', variables=['offl_eta','offl_pt', 'rho', 'npv']):
    # Retrieve data from tree
    branches = [iso]
    branches.extend(variables)
    data = root2array(inputfile, treename=tree, branches=branches, selection='et>0')
    data = data.view((np.float64, len(data.dtype.names)))
    iso = data[:, [0]].astype(np.int32).ravel()
    graphs = []
    # Compute efficiencies for each variable
    for i,variable in enumerate(variables):
        xs  = data[:, [1+i]].astype(np.float32).ravel()
        graphs.append(efficiency.efficiency_graph(pass_function=(lambda x: x>0), function_inputs=iso, xs=xs))
        graphs[-1].SetName('current_'+variable+'_test')
    return graphs

def test_isolation_workingpoints(effs, isolations, inputfile, tree, inputnames=['abs(ieta)','ntt'], targetname='iso', variables=['offl_eta','offl_pt', 'rho', 'npv']):
    # Retrieve data from tree
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    branches.extend(variables)
    data = root2array(inputfile, treename=tree, branches=branches, selection='et>0')
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
    #print signal_efficiencies_diff
    signal_efficiencies_diff_3 = np.convolve(signal_efficiencies_diff, np.repeat(1.0, 3.)/3., 'valid')
    signal_efficiencies_diff_2 = np.convolve(signal_efficiencies_diff, np.repeat(1.0, 2.)/2., 'valid')
    signal_efficiencies_diff = np.append(signal_efficiencies_diff_3, [signal_efficiencies_diff_2[-1],signal_efficiencies_diff[-1]])
    #print signal_efficiencies_diff
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


def optimize_background_rejection_vs_ieta(effs, isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputnames=['abs(ieta)','ntt'], targetname='iso', cut='et>10'):
    #ieta_binning = np.arange(0.5,28.5,1)
    #ieta_binning = [0.5, 3.5, 6.5, 9.5, 13.5, 18.5, 22.5, 27.5]
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

def relax_efficiency_vs_pt_2(optimal_points_vs_ieta_low, optimal_points_vs_ieta_high, threshold_low, threshold_high, eff_min=0.4,max_et=110):
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

def isolations_vs_threshold(effs, isolations, optimal_points):
    isolations_relaxed = {}
    for threshold in range(20, 108, 2):
        points_vs_pt_ieta = relax_efficiency_vs_pt(optimal_points, threshold)
        combined_cuts_pt = CombinedWorkingPoints(effs, isolations, points_vs_pt_ieta)
        isolations_relaxed[threshold] = combined_cuts_pt
    return isolations_relaxed

def rate_optimal(thresholds_isolations, inputfile, tree, inputnames=['abs(ieta)','ntt','et_raw'], targetname='iso', rate_constraint=10.):
    # Retrieve data from tree
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    branches.extend(['Run', 'Event'])
    data = root2array(inputfile, treename=tree, branches=branches, selection='et>0')
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    et = data[:, [ninputs-1]].astype(np.float32).ravel()
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    run_events  = data[:, [ninputs+1,ninputs+2]].astype(np.int32)
    # Count total number of unique run/event pairs
    indices_sort = np.lexsort(run_events.T);
    total_events = len(run_events[np.concatenate(([True],np.any(run_events[indices_sort[1:]]!=run_events[indices_sort[:-1]],axis=1)))])
    for threshold,isolation in sorted(thresholds_isolations.items()):
        print threshold
        et_mask = np.argwhere(et>=threshold).ravel()
        #print et_mask
        et_cut = et[et_mask].ravel()
        inputs_cut = inputs[et_mask]
        targets_cut = targets[et_mask].ravel()
        run_events_cut = run_events[et_mask]
        #print len(et_cut)
        #print et_cut
        #print inputs_cut
        #print isolation.value(inputs_cut[:,[0,1]],inputs_cut[:,[0,2]])
        #print np.less(targets_cut,isolation.value(inputs_cut[:,[0,1]],inputs_cut[:,[0,2]]))
        iso_mask = np.argwhere(np.less(targets_cut,isolation.value(inputs_cut[:,[0,1]],inputs_cut[:,[0,2]]))).ravel()
        #print iso_mask
        #print len(iso_mask)
        print rate.rate(et_cut, run_events_cut, [threshold], mask=iso_mask, total_events=total_events)

def rates(isolation, inputfile, tree, inputnames=['abs(ieta)','ntt','et_raw'], targetname='iso', ref_iso='iso_pass'):
    # Retrieve data from tree
    print '1'
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    branches.append(ref_iso)
    branches.append('et')
    branches.extend(['Run', 'Event'])
    data = root2array(inputfile, treename=tree, branches=branches, selection='et>0')
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    pass_ref_iso  = data[:, [ninputs+1]].astype(np.int32).ravel()
    et = data[:, [ninputs+2]].astype(np.float32).ravel()
    run_events  = data[:, [ninputs+3,ninputs+4]].astype(np.int32)
    # Count total number of unique run/event pairs
    indices_sort = np.lexsort(run_events.T);
    total_events = len(run_events[np.concatenate(([True],np.any(run_events[indices_sort[1:]]!=run_events[indices_sort[:-1]],axis=1)))])
    pass_indices = np.argwhere(et >= 20).ravel()
    inputs = inputs[pass_indices]
    et = et[pass_indices]
    targets = targets[pass_indices]
    pass_ref_iso = pass_ref_iso[pass_indices]
    run_events = run_events[pass_indices]
    thresholds = np.arange(20., 100., 2.)
    iso_mask = np.argwhere(np.less(targets,isolation.value(inputs[:,[0,1]],inputs[:,[0,2]]))).ravel()
    iso_current_mask = np.argwhere(pass_ref_iso>0).ravel()
    rates_new = rate.rate(et, run_events, thresholds, mask=iso_mask, total_events=total_events)
    rates_current = rate.rate(et, run_events, thresholds, mask=iso_current_mask, total_events=total_events)
    rates_noiso = rate.rate(et, run_events, thresholds, mask=None, total_events=total_events)
    return np.column_stack((rates_current, rates_new[:,[1]].ravel(), rates_noiso[:,[1]].ravel()))

def rates_fromhisto(isolation, inputfile, tree, inputnames=['abs(ieta)','ntt','et_raw'], targetname='iso', ref_iso='iso_pass'):
    # Retrieve data from tree
    print '1'
    ninputs = len(inputnames)
    branches = copy.deepcopy(inputnames)
    branches.append(targetname)
    branches.append(ref_iso)
    branches.append('et')
    branches.extend(['Run', 'Event'])
    data = root2array(inputfile, treename=tree, branches=branches, selection='et>0')
    data = data.view((np.float64, len(data.dtype.names)))
    inputs = data[:, range(ninputs)].astype(np.float32)
    targets  = data[:, [ninputs]].astype(np.float32).ravel()
    pass_ref_iso  = data[:, [ninputs+1]].astype(np.int32).ravel()
    et = data[:, [ninputs+2]].astype(np.float32).ravel()
    run_events  = data[:, [ninputs+3,ninputs+4]].astype(np.int32)
    # Count total number of unique run/event pairs
    indices_sort = np.lexsort(run_events.T);
    total_events = len(run_events[np.concatenate(([True],np.any(run_events[indices_sort[1:]]!=run_events[indices_sort[:-1]],axis=1)))])
    pass_indices = np.argwhere(et >= 20).ravel()
    inputs = inputs[pass_indices]
    et = et[pass_indices]
    targets = targets[pass_indices]
    pass_ref_iso = pass_ref_iso[pass_indices]
    run_events = run_events[pass_indices]
    thresholds = np.arange(20., 200., 2.)
    iso_mask = np.argwhere(np.less(targets,evaluate(isolation, inputs))).ravel()
    iso_current_mask = np.argwhere(pass_ref_iso>0).ravel()
    rates_new = rate.rate(et, run_events, thresholds, mask=iso_mask, total_events=total_events)
    rates_current = rate.rate(et, run_events, thresholds, mask=iso_current_mask, total_events=total_events)
    rates_noiso = rate.rate(et, run_events, thresholds, mask=None, total_events=total_events)
    return np.column_stack((rates_current, rates_new[:,[1]].ravel(), rates_noiso[:,[1]].ravel()))

def main(signalfile, signaltree, backgroundfile, backgroundtree, outputdir, name, test=False, inputs=['abs(ieta)','ntt'], target='iso', pileupref='rho'):
    # Compute isolation cuts for efficiencies from 0.2 to 1 with smaller steps for larger efficiencies
    effs = np.arange(0.2,0.5,0.05)
    effs = np.concatenate((effs,np.arange(0.5,0.85,0.02)))
    effs = np.concatenate((effs,np.arange(0.85,0.999,0.01)))
    #effs = np.arange(0.6,1.,0.1) # for tests
    version = batch_launcher.job_version(outputdir)
    #version = 'v_5_2016-07-13' # RunC
    #version = 'v_6_2016-07-19' # RunD
    #version = 'v_7_2016-07-21' # V3
    #version = 'v_8_2016-07-29' # V3 with et_raw fix
    version = 'v_10_2016-08-01'
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
        #print '> Optimizing signal efficiency vs background rejection'
        ##for et_cut in [10, 15, 20, 30, 40, 50]:
        #et_cut = 20
        #signal_efficiencies_diff_graph, background_efficiencies_diff_graph, optimal_points_graph, optimal_point = optimize_background_rejection(effs, eg_isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputs, target, cut='et>{}'.format(et_cut))
        #print '   Best inclusive working point', optimal_point
        #signal_efficiencies_diff_graph.SetName(signal_efficiencies_diff_graph.GetName()+'_et_{}'.format(et_cut))
        #background_efficiencies_diff_graph.SetName(background_efficiencies_diff_graph.GetName()+'_et_{}'.format(et_cut))
        #optimal_points_graph.SetName(optimal_points_graph.GetName()+'_et_{}'.format(et_cut))
        #signal_efficiencies_diff_graph.Write() 
        #background_efficiencies_diff_graph.Write()
        #optimal_points_graph.Write()
        #signal_efficiencies_diff_graphs, background_efficiencies_diff_graphs, optimal_points_graphs, optimal_points = optimize_background_rejection_vs_ieta(effs, eg_isolations, signalfile, signaltree, backgroundfile, backgroundtree, inputs, target, cut='et>{}'.format(et_cut))
        #print '   Best working points vs |ieta|', hist2array(optimal_points)
        #for graph in signal_efficiencies_diff_graphs: 
            #graph.Write()
        #for graph in background_efficiencies_diff_graphs:
            #graph.Write()
        #for graph in optimal_points_graphs:
            #graph.Write()
        #optimal_points.SetName('optimal_points_vs_ieta')
        #optimal_points.Write()
        #optimal_point = 0.824074074074 
        #optimal_points_array = np.array([0.85000002,0.92777777,0.94629627,0.93222225,0.84851849,0.93444443,0.56777775])
        #optimal_points_array = np.array([0.85000002,0.92777777,0.94629627,0.93222225,0.90,0.85,0.56777775])
        #optimal_points_array = np.array([0.93000002,0.92777777,0.94629627,0.93222225,0.93,0.85,0.56777775])
        # relaxed endcaps and center
        #optimal_points_array = np.array([0.93000002,0.92777777,0.94629627,0.93222225,0.93,0.85,0.85, 0.70])
        # relaxed endcaps and cut barrel
        optimal_points_array = np.array([0.85,0.85,0.89,0.89,0.87,0.80,0.80, 0.70])
        ieta_binning = [0.5, 3.5, 6.5, 9.5, 13.5, 18.5, 22.5, 25.5, 28.5]
        optimal_points = Hist(ieta_binning)
        array2hist(optimal_points_array, optimal_points)
        ################
        optimal_points_low_array = np.array([0.80,0.80,0.80,0.80,0.80,0.75,0.80, 0.85])
        optimal_points_high_array = np.array([0.92,0.95,0.95,0.95,0.95,0.95,0.95, 0.95])
        ieta_binning = [0.5, 3.5, 6.5, 9.5, 13.5, 18.5, 22.5, 25.5, 28.5]
        optimal_points_low = Hist(ieta_binning)
        optimal_points_high = Hist(ieta_binning)
        array2hist(optimal_points_low_array, optimal_points_low)
        array2hist(optimal_points_high_array, optimal_points_high)
        ################
        combined_cuts = CombinedWorkingPoints(effs, [iso.predict for iso in eg_isolations], optimal_points)
        #combined_cuts_histo = function2th2(lambda x: combined_cuts.value(x,x[:,[0]].ravel()), quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]])
        combined_cuts_histo = function2th2(lambda x: combined_cuts.value(x,x[:,[0]]), quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]])
        print '> Applying pt relaxation'
        #points_vs_pt_ieta = relax_efficiency_vs_pt(optimal_points, 60)
        #combined_cuts_pt = IsolationCombinedCuts(np.append(effs,[1.]), [iso.predict for iso in eg_isolations]+[lambda x:np.full(x.shape[0],9999.)], points_vs_pt_ieta)
        #print '> Converting pt relaxed into histo'
        #combined_cuts_pt_histo = function2th3(lambda x: combined_cuts_pt.value(x[:,[0,1]],x[:,[0,2]]), quantile_regression.binning[inputs[0]], quantile_regression.binning[inputs[1]], quantile_regression.binning['et'])
        #eg_isolations_relaxed = isolations_vs_threshold(np.append(effs,[1.]), [iso.predict for iso in eg_isolations]+[lambda x:np.full(x.shape[0],9999.)], optimal_points)
        #points_vs_pt_ieta_thomas_list = [relax_efficiency_vs_pt(optimal_points, threshold=th) for th in [40.,50.,60.,70.]]
        points_vs_pt_ieta_thomas_list = [relax_efficiency_vs_pt_2(optimal_points_low, optimal_points_high, threshold_low=56., threshold_high=80., eff_min=0.5, max_et=max_et) for max_et in [110.,120.,130.]]
        eg_isolation_relaxed_thomas_list = [CombinedWorkingPoints(np.append(effs,[1.]), [iso.predict for iso in eg_isolations]+[lambda x:np.full(x.shape[0],9999.)], points_vs_pt_ieta_thomas) for points_vs_pt_ieta_thomas in points_vs_pt_ieta_thomas_list]
        print '> Testing combined iso cuts'
        #graphs_comb = test_combined_isolation(combined_cuts_histo, signalfile, signaltree, inputs, target)
        #graphs_comb = test_combined_isolation(combined_cuts, signalfile, signaltree, inputs, target)
        #for graph in graphs_comb:
            #graph.Write()
        print '> Testing combined iso cuts vs pt'
        ###rate_optimal(eg_isolations_relaxed, backgroundfile, backgroundtree)
        graphs_iso_current = test_current_isolation(signalfile, signaltree)
        for graph in graphs_iso_current:
            graph.Write()
        #for i,eg_isolation_relaxed_thomas in enumerate(eg_isolation_relaxed_thomas_list):
            #print i
            #rates(eg_isolation_relaxed_thomas, backgroundfile, backgroundtree)
            #graphs_comb_pt_thomas = test_combined_isolation_pt(eg_isolation_relaxed_thomas, signalfile, signaltree, inputs+['et_raw'], target, variables=['offl_eta','offl_pt', 'rho'])
            #for graph in graphs_comb_pt_thomas:
                #graph.SetName(graph.GetName()+'_thomas_{}'.format(i))
                #graph.Write()
        print '> Compress isolation tables'
        # Retrieve data from tree
        et_compress_bins = [0,18,20,22,28,32,37,42,52,63,73,81,87,91,111,151,256] 
        ieta_compress_bins = [0,5,6,9,10,12,13,14,17,18,19,20,23,24,25,26,32]
        ntt_compress_bins = [0,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86,91,96,101,106,111,116,121,126,131,136,141,146,151,156,256]
        branches = ['abs(ieta)','ntt','et_raw']
        data = root2array(signalfile, treename=signaltree, branches=branches, selection='et>0')
        data = data.view((np.float64, len(data.dtype.names))).astype(np.float32)
        for i,eg_isolation_relaxed_thomas in enumerate(eg_isolation_relaxed_thomas_list):
            print i
            iso_cuts = eg_isolation_relaxed_thomas.value(data[:,[0,1]],data[:,[0,2]])
            eg_isolation_compressed = events2th3(data, iso_cuts, (ieta_compress_bins,), (ntt_compress_bins,), (et_compress_bins,))
            eg_isolation_compressed.SetName('isolation_compressed_{}'.format(i))
            eg_isolation_compressed.Write()
            graphs_compressed = test_combined_isolation_pt_compressed(eg_isolation_compressed, signalfile, signaltree, inputs+['et_raw'], target, variables=['offl_eta','offl_pt', 'rho'])
            for graph in graphs_compressed:
                graph.SetName(graph.GetName()+'_{}'.format(i))
                graph.Write()
            rates_compressed = rates_fromhisto(eg_isolation_compressed, backgroundfile, backgroundtree)
            if i==0:
                rate_current = Graph(len(rates_compressed))
                rate_noiso = Graph(len(rates_compressed))
                fill_graph(rate_current, rates_compressed[:,[0,1]])
                fill_graph(rate_noiso, rates_compressed[:,[0,3]])
                rate_current.SetName('rate_compressed_current')
                rate_noiso.SetName('rate_compressed_noiso')
                rate_current.Write()
                rate_noiso.Write()
            rate_new = Graph(len(rates_compressed))
            fill_graph(rate_new, rates_compressed[:,[0,2]])
            rate_new.SetName('rate_compressed_new_{}'.format(i))
            rate_new.Write()
        #for threshold in [20, 50, 100]:
            #print '  Threshold', threshold
            #combined_cuts_pt = eg_isolations_relaxed[threshold]
            #graphs_comb_pt = test_combined_isolation_pt(combined_cuts_pt, signalfile, signaltree, inputs+['et'], target)
            #for graph in graphs_comb_pt:
                #graph.SetName(graph.GetName()+'_threshold_{}'.format(threshold))
                #graph.Write()
        #combined_cuts_histo.SetName('optimal_cuts')
        #combined_cuts_histo.Write()

        #points_vs_pt_ieta.SetName('points_vs_pt_ieta')
        #points_vs_pt_ieta.Write()



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

