<table>
  <tr>
    <td width="250" height="250"><img src="https://github.com/AntonioMG99/IPTP-paper-code/assets/52926867/bb92795a-7016-4814-98dd-833bc9a38e5d" alt="logo" /></td>
    <!-- Use span with styles to mimic h1 -->
    <td><h1>IPTP-paper-code: Inverse Problems in Turing Patterns</h1> Code for the paper: "Unraveling biochemical spatial patterns: machine learning approaches to the inverse problem of Turing patterns" </td>
  </tr>
</table>
## Setup

* Tested on Mac OS Ventura 13.3.1(a) and Rocky Linux

* Requirements:\
  Python 3.9.16\
  Tensorflow 2.11.0\
  Numpy 1.22.1\
  Maptlotlib \
  SciPy 1.9.3 \
  opencv-python 4.9.0.80\
  shapely 2.0.3

## Structure of code

Code is divided into three main files, found in the folder 'models':

* The `Least_Squares.ipynb` file contains all code involving the Least Squares method and plots for the figures in the paper. This is uploaded as a Jupyter Notebook since the code is relatively fast and many results are shown. 

* The `RBFPINNs.py` file contains all code involving the RBF-PINNs method applied to the PDE models discussed in the paper (Schnakenberg, FitzHugh-Nagumo and Brusselator) and plots for the figures. 

* The `RBFPINNs_ChemPat.py` file contains all code involving the RBF-PINNs method applied to the chemical patterns and plots for the figures.

## Utils

Inside the folder 'utils', there is the 'Cyclic Learning rate' callback function, which has been adapted from https://github.com/bckenstler/CLR. This is a learning rate scheduler found helpful for a faster convergence of the neural networks.

## Datasets

Inside the datasets folder, there are the perturbation arrays containing the initial conditions to reproduce the results in the paper, and the chemical patterns data from the paper 'Turing patterns on radially growing domains: experiments and simulations' available in https://doi.org/10.1039/C8CP07797E.
