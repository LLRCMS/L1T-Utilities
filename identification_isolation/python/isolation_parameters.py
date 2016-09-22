

class IsolationSteps:
    def __init__(self):
        self.train_workingpoints = True
        #self.test_overtraining = False
        self.fit_ntt_vs_rho = True
        self.test_workingpoints = True
        self.do_compression = True

class IsolationVariables:
    def __init__(self):
        self.ieta = 'ieta'
        self.et = 'et'
        self.ntt = 'ntt'
        self.rho = 'rho'
        self.iso = 'iso'

class IsolationOptimization:
    def __init__(self):
        self.eta_optimization = 'none'
        self.eta_pt_efficiency_shapes = None

class IsolationCompression:
    def __init__(self):
        self.eta = [0,5,6,9,10,12,13,14,17,18,19,20,23,24,25,26,32]
        self.et = [0,18,20,22,28,32,37,42,52,63,73,81,87,91,111,151,256]
        self.ntt = [0,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,86,91,96,101,106,111,116,121,126,131,136,141,146,151,156,256]

class IsolationParameters:
    def __init__(self):
        self.name = ''
        self.version = 'automatic'
        self.signal_file = ''
        self.signal_tree = ''
        self.background_file = ''
        self.background_tree = ''
        self.working_directory = ''
        self.steps = IsolationSteps()
        self.variables = IsolationVariables()
        self.eta_pt_optimization = IsolationOptimization()
        self.compression = IsolationCompression()

