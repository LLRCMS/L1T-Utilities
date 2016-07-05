import copy
import numpy as np

parameters_template = {
    'inputfile':'/data_CMS/cms/sauvan/L1/2016/IsolationValidation/ZElectron/v_3_2016-06-23/tagAndProbe_isolationValidation_2016B_ZElectron.root',
    'tree':'ntTagAndProbe_IsolationValidation_Stage2_Rebuilt_tree',
    'inputs':'"abs(ieta), rho"',
    'target':'iso',
    'outputfile':'test.root',
    'name':'iso',
    'eff':'0.6',
    'test':'',
}


# Define sets of parameters with different efficiency working points
# from 0.5 to 1 with steps of 0.1
parameters = []
for eff in np.arange(0.5, 1., 0.1):
    parameters.append(copy.deepcopy(parameters_template))
    parameters[-1]['eff'] = str(eff)
    parameters[-1]['outputfile'] = 'test_iso_eff_{}.root'.format(eff*100)
