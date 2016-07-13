# Installation and setup
This has been tested on LLR machines (polui, llrui)  

## Initial install
```
git clone git@github.com:LLRCMS/L1T-Utilities.git
cd L1T-Utilities 
git remote add llrcms git@github.com:LLRCMS/L1T-Utilities.git
git checkout -b <my-devel-branch>
source initialize 
```

This last command will:
* initialize a CMSSW area in `./CMSSW/`
* Install python `pip` utility in `~/.local/`
* Install and intialize a python virtual environment (`virtualenv`) in `./env/`
* Install needed python dependencies
  * `SciPy` and `NumPy`
  * `scikit-learn`
  * `root_numpy`
  * `rootpy`

## Environment setup
`source setupenv`  
It will setup the CMSSW environment and activate the python virtual environment  

# Available utilities
## `batch`
A script used to send jobs on the LLR t3 is available (`python/batch_launcher.py`). It is used in the following way:
```
Usage: python batch_launcher.py [options]

Options:
  -h, --help               show this help message and exit
  --workingdir=WORKING_DIR Working directory, where the jobs will be run. It will be created if it
                           doesn't exist
  --exe=EXECUTABLE         Executable (e.g. "python quantile_regression.py")
  --pars=PARAMETER_FILE    Python file containing the list of parameters
Optional options:
  --name=NAME              Name of the batch jobs
  --queue=QUEUE            Batch queue
  --proxy=PROXY            Grid user proxy
```
This command needs to be launched from the directory where the python parameter file is. Otherwise it won't be able to import the parameters.   

The parameter file is a python file that must contain a variable `parameters`, which must be a list of dictionaries. Each item in the list corresponds to one job, and the dictionary contains the set of parameters to pass to the executable for this job. There is an exemple in `identification_isolation/config/test_iso_batch.py`.   

The jobs have a version attached to them, which is automatically computed. When jobs are sent in a given `workingdir`, a version number will be associated to them and a subdirectory will be created such that they don't interfere with the previous results produced in the same `workingdir`.


## `identification_isolation`

### `python/egamma_isolation`
This script runs all the steps needed for the egamma isolation:
* Perform a linear regression of `ntt` vs `rho`
* Train multiple quantile regressions to derive isolation cuts for several efficiency working points. These cuts are derived as a function of `|ieta|` and `rho`. The trainings will be launched on batch.
* Apply the `ntt` to `rho` mapping to the isolation regression and save results as 2D histograms (`|ieta|`, `ntt`) for all the working points.   
* Produce efficiencies of all the working points vs eta and pt of the offline electron, and npv, rho
* Find the optimal inclusive working point, in terms of background rejection and signal efficiency
* Find the optimal working point in bins of |ieta|

The optimization of the working points is done by looking at the efficiency gradient for signal and background. The optimal working point is chosen as the point where the background gradient becomes smaller or equal to the signal gradient. This means that cutting harder than this point will kill signal more (or equally) than it kills background.   

```
Usage: python egamma_isolation.py [options]

Options:
  -h, --help              show this help message and exit
  --inputfile=INPUT_FILE  Input file
  --tree=TREE_NAME        Tree in the input file
  --outputdir=OUTPUT_DIR  Output directory
  --name=NAME             Name used for the results
  --test                  Flag to test regression on a test sample
  --inputs=INPUTS         List of input variables of the form "var1,var2,..."
  --pileupref=PILEUP_REF  Reference variable used for pile-up
  --target=TARGET         Target variable
```


### `python/quantile_regression`
This script computes cuts to be applied on a target variable, function of several input variables. These cuts are determined such that they give a flat efficiency as a function of the input variables.  

The training of one single working point can be done with the following command:
```
Usage: python quantile_regression.py [options]
Options:
  -h, --help                Show this help message and exit
  --inputfile=INPUT_FILE    Input file
  --tree=TREE_NAME          Tree in the input file
  --inputs=INPUTS           List of input variables of the form "var1,var2,..."
  --target=TARGET           Target variable
  --outputfile=OUTPUT_FILE  Output file
  --name=NAME               Name used to store the regression results in the output file
  --test                    Flag to test the regression on a test sample
```
If two or three input variables are used the regression result will be stored inside a 2D or 3D histogram in the output file. In addition the scikit-learn object will be stored in a pickle file.

### `python/correlations`
Currently this module provides only a linear regression beween two variables, used to correlate two pile-up variables (such as `ntt` and `rho`).

```
Usage: python correlations.py [options]

Options:
  -h, --help              show this help message and exit
  --inputfile=INPUT_FILE  Input file
  --tree=TREE_NAME        Tree in the input file
  --input=INPUT           Input variable
  --target=TARGET         Target variable
  --test                  Flag to test regression on a test sample
```

### `python/efficiency`
`efficiency_inclusive` produces the inclusive efficiency from a selection function.
```
Parameters:
	pass_function: function used to select events. Must be able to take arrays as inputs
    function_inputs: inputs given to the function
Returns:
    Efficiency, down error, up error
```

`efficiency_graph` produces a 1D efficiency graph from a selection function.
```
Parameters:
	pass_function: function used to select events. Must be able to take arrays as inputs
    function_inputs: inputs given to the function
    xs: values of the variable of interest, used to fill the histograms
    bins: histogram binning. If no binning is given, it will be computed automatically
    error: target uncertainty on the efficiciency, used to determine the automatic binning
Returns:
    Efficiency graph
```



