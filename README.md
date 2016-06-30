# Installation and setup
This has been tested on LLR machines (polui, llrui)

## Initial install
`git clone git@github.com:LLRCMS/L1T-Utilities.git`
`cd L1T-Utilities`
`source initialize`

This last command will:
* initialize a CMSSW area in `./CMSSW/`
* Install python `pip` utility in ~/.local/
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


