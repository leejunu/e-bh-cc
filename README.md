# e-bh-cc

Documentation soon!

## Setting up the virtual environments

The simulation scripts require certain Python libraries to run.
Furthermore, specific instantiations of e-BH-CC require more
specific machinery. We will describe how to create virtual
environments for each of the three settings with the needed
imports.

### CC for z-testing and t-testing

Create the Python3 virtual environment (venv) by running the following
terminal command:
```
python3 -m venv venv_ebhcc
```
To activate the venv and install the required dependencies for the z-test
and t-test scripts, run in the terminal
```
source venv_ebhcc/bin/activate
pip install -r requirements.txt
```
In this virtual environment, you can now run the z-testing and t-testing
experiments (e.g. `ztesting_CC.py` and `ttesting_CC.py`).

### CC for knockoffs

Create the Python3 virtual environment (venv) by running the following
terminal command:
```
python3 -m venv venv_kn_mvr
```
To activate the venv and install the required dependencies for the 
knockoffs scripts, run in the terminal
```
source venv_kn_mvr/bin/activate
pip install -r req_knockoffs.txt
# need the choldate package for MVR knockoffs
pip install git+https://github.com/jcrudy/choldate.git@d37246f4fc1775f11b84d42b5ceba08e6392d285  
```
In this virtual environment, you can now run the model-X knockoffs 
experiments, which use the `mxknockoffs_CC.py` file.

_(Note: we will add instructions on how to use SDP knockoffs at a later point. 
These require a different set of Python dependencies.)_

### CC for conformalized outlier detection

Create the Python3 virtual environment (venv) by running the following
terminal command:
```
python3 -m venv venv_numba
```
This is named due to its usage of the `numba` JIT compiler, which makes
the `numpy` operations required in our implementation of conformal 
selection extremely fast.


To activate the venv and install the required dependencies for conformal
outlier detection scripts, run in the terminal
```
source venv_numba/bin/activate
pip install -r req_numba.txt
```
In this virtual environment, you can now run outlier detection experiments,
which use the `outlier_detection_CC.py` file.
