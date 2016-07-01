# Installation and setup
This has been tested on LLR machines (polui, llrui)  

## Initial install
`git clone git@github.com:LLRCMS/L1T-Utilities.git`  
`cd L1T-Utilities`  
`source initialize`  

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
## `identification_isolation`
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


