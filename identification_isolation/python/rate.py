import copy
import numpy as np
from root_numpy import root2array
from rootpy.plotting import Hist

nbunches = 2064.
c_light = 299792.458
lhc_length = 27.
rate_per_bunch = 11245. 

def rate(et_values, event_runs, et_cuts, mask=None, total_events=0):
    et_values_copy = copy.deepcopy(et_values)
    event_runs_copy = copy.deepcopy(event_runs)
    if total_events==0:
        # Count number of unique run/event pairs
        indices_sort = np.lexsort(event_runs_copy.T);
        total_events = len(event_runs_copy[np.concatenate(([True],np.any(event_runs_copy[indices_sort[1:]]!=event_runs_copy[indices_sort[:-1]],axis=1)))])
    # Apply mask
    if mask!=None:
        et_values_copy = et_values_copy[mask]
        event_runs_copy = event_runs_copy[mask]
    # Make sure the et thresholds are sorted
    et_cuts_sorted = np.sort(et_cuts)
    rates = []
    for cut in et_cuts_sorted:
        pass_indices = np.argwhere(et_values_copy >= cut).ravel()
        et_values_copy = et_values_copy[pass_indices]
        event_runs_copy = event_runs_copy[pass_indices]
        indices_sort = np.lexsort(event_runs_copy.T);
        pass_events = len(event_runs_copy[np.concatenate(([True],np.any(event_runs_copy[indices_sort[1:]]!=event_runs_copy[indices_sort[:-1]],axis=1)))])
        #rates.append([cut, float(pass_events)/float(total_events)*nbunches*c_light/lhc_length*1.e-3])
        #rates.append([cut, float(pass_events)/float(total_events)*nbunches*rate_per_bunch*1.e-3])
        rates.append([cut, float(pass_events)/float(total_events)])
    return np.array(rates, dtype=np.float64)

def rate_test(et_values, event_runs, et_cuts):
    events = {}
    for et,(run_event) in zip(et_values, event_runs):
        event = tuple(run_event)
        if event in events:
            if et>events[event]: events[event] = et
        else: events[event] = et
    histo = Hist(256, -0.5, 255.5)
    for event,et in events.items():
        histo.Fill(et)
    total_events = histo.integral(overflow=True)
    rates = []
    for et_cut in et_cuts: 
        bin = histo.GetXaxis().FindBin(et_cut)
        pass_events = histo.integral(bin, overflow=True)
        rates.append([et_cut, float(pass_events)/float(total_events)*nbunches*c_light/lhc_length*1.e-3])
    return np.array(rates, dtype=np.float64)



if __name__=='__main__':
    file_name = '/data_CMS/cms/sauvan/L1/2016/IsolationNtuples/ZeroBias_2016C_1e34/v_2_2016-07-19/zeroBias_IsolationNtuple.root'
    tree_name = 'ntZeroBias_IsolationNtuple_tree'
    branches = ['et', 'Run', 'Event']
    data = root2array(file_name, treename=tree_name, branches=branches)
    data = data.view((np.float64, len(data.dtype.names)))
    et_values = data[:, [0]].ravel()
    run_events_raw  = data[:, [1,2]]
    run_events  = data[:, [1,2]].astype(np.int32)
    #pass_run = np.where(run_events[:,[0]].ravel()==276352)
    #print pass_run
    #run_events = run_events[pass_run]
    #et_values = et_values[pass_run]
    #print run_events
    a = rate(et_values,run_events, np.arange(0., 200., 1.) )
    #b =  rate_test(et_values,run_events, np.arange(10., 200., 1.))
    #print np.column_stack((a,b))
    print a

