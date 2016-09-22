import numpy as np
from identification_isolation.isolation_parameters import IsolationParameters
from identification_isolation.egamma_isolation import double_slope_relaxation_vs_pt

from root_numpy import array2hist
from rootpy.plotting import Hist


parameters = IsolationParameters()
## General
parameters.name = 'isolation_egamma'
parameters.version = 'automatic'
parameters.signal_file = '/data_CMS/cms/sauvan/L1/2016/V3_new_retunedSK/IsolationValidation/2016C/ZElectron//v_1_2016-08-05/tagAndProbe_isolationValidation_2016C_ZElectron.root'
parameters.signal_tree = 'ntTagAndProbe_IsolationValidation_Stage2_Rebuilt_tree'
parameters.background_file = '/data_CMS/cms/sauvan/L1/2016/V3_new_retunedSK/IsolationNtuples/ZeroBias_2016C_1e34/v_1_2016-08-06/zeroBias_IsolationNtuple.root'
parameters.background_tree = 'ntZeroBias_IsolationNtuple_tree'
parameters.working_directory = '/home/llr/cms/sauvan/DATA/TMP/egamma_isolation_retunedSK/'
## Variable names
parameters.variables.ieta = 'abs(ieta)'
parameters.variables.et = 'et_raw'
parameters.variables.ntt = 'ntt'
parameters.variables.rho = 'rho'
parameters.variables.iso = 'iso'
## Steps
parameters.steps.train_workingpoints = True
parameters.steps.fit_ntt_vs_rho = True
parameters.steps.test_workingpoints = True
parameters.steps.do_compression = True
## eta-pt efficiency shape
parameters.eta_pt_optimization.eta_optimization = 'none'
efficiencies_low_array = np.array([0.80,0.80,0.80,0.80,0.80,0.75,0.80, 0.85])
efficiencies_high_array = np.array([0.92,0.95,0.95,0.95,0.95,0.95,0.95, 0.95])
eta_binning = [0.5, 3.5, 6.5, 9.5, 13.5, 18.5, 22.5, 25.5, 28.5]
efficiencies_low = Hist(eta_binning)
efficiencies_high = Hist(eta_binning)
array2hist(efficiencies_low_array, efficiencies_low)
array2hist(efficiencies_high_array, efficiencies_high)
parameters.eta_pt_optimization.eta_pt_efficiency_shapes = \
        double_slope_relaxation_vs_pt(efficiencies_low,\
                                      efficiencies_high,\
                                      threshold_low=56.,\
                                      threshold_high=80.,\
                                      eff_min=0.5,\
                                      max_et=120.)
## LUT Compression
parameters.compression.eta = [0,5,6,9,10,12,13,14,17,18,19,20,23,24,25,26,32]
parameters.compression.et = [0,18,20,22,28,32,37,42,52,63,73,81,87,91,111,151,256]
parameters.compression.ntt = [0,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86,91,96,101,106,111,116,121,126,131,136,141,146,151,156,256]
